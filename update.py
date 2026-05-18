import copy
import math
import os

import models
import torch
from poison import add_trigger, add_trigger_batch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from utils import project_within_radius


adversary_list = [0, 1, 2]
poison_epoch = {
    0: range(12, 113),
    1: range(12, 113),
    2: range(12, 113),
}
poisoning_per_batch = 8
scale_weights = 1
poison_label_swap = 2


class TargetNeuronActivation:

    def __init__(self, target_neuton_index):
        self.activation = None
        self.target_neuron_index = target_neuton_index

    def __call__(self, module, input, output):
        self.activation = output


class DatasetSplit(Dataset):

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        if torch.is_tensor(image):
            image_tensor = image
        else:
            image_tensor = torch.as_tensor(image)
        return image_tensor, torch.as_tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger, idx, layer_neuron_indices, global_round, trigger_value):
        self.args = args
        self.logger = logger
        self.idx = idx
        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.layer_neuron_indices = layer_neuron_indices
        self.global_round = global_round
        self.trigger_value = trigger_value

    def train_val_test(self, dataset, idxs):

        idxs_train = idxs[:int(0.8 * len(idxs))]
        idxs_val = idxs[int(0.8 * len(idxs)):int(0.9 * len(idxs))]
        idxs_test = idxs[int(0.9 * len(idxs)):]

        num_workers = min(2, os.cpu_count() or 0)
        loader_kwargs = {
            'num_workers': num_workers,
            'pin_memory': self.device.type == 'cuda',
        }

        trainloader = DataLoader(
            DatasetSplit(dataset, idxs_train),
            batch_size=self.args.local_bs,
            shuffle=True,
            drop_last=True,
            **loader_kwargs,
        )
        validloader = DataLoader(
            DatasetSplit(dataset, idxs_val),
            batch_size=int(len(idxs_val) / 10),
            shuffle=False,
            drop_last=True,
            **loader_kwargs,
        )
        testloader = DataLoader(
            DatasetSplit(dataset, idxs_test),
            batch_size=int(len(idxs_test) / 10),
            shuffle=False,
            drop_last=True,
            **loader_kwargs,
        )
        return trainloader, validloader, testloader

    def update_weights(self, model):
        target_model = None
        adversarial_index = -1
        model.train()
        epoch_loss = []
        local_ep = self.args.local_ep
        generate_trigger = False

        def selective_grad_update_hook(module, grad_input, grad_output):
            module_name = None
            for name, m in model.named_modules():
                print(name)
                if m is module:
                    module_name = name
                    print(module_name)
                    break
                else:
                    print('not')

            if module_name in self.layer_neuron_indices.key():
                indices = self.layer_neuron_indices[module_name]

                if isinstance(module, nn.Conv2d) and grad_input[0] is not None:
                    mask = torch.zeros_like(grad_input[0])

                    for idx in indices:
                        if len(idx) == 4:
                            oc, ic, kh, kw = idx
                            mask[oc, ic, kh, kw] = 1

                    modified_grad = grad_input[0] * mask
                    return (modified_grad,) + grad_input[1:]

                elif isinstance(module, nn.Linear) and grad_input[0] is not None:
                    pass

            return grad_input * 0

        lr = 0.5 * self.args.lr * (1 + math.cos(math.pi * self.global_round / self.args.epochs))

        if self.idx in adversary_list:
            for temp_index in range(0, len(adversary_list)):
                if int(self.idx) == adversary_list[temp_index]:
                    adversarial_index = temp_index % 3
                    break

        if (self.idx in adversary_list) and (self.global_round in poison_epoch[adversarial_index]):
            local_ep = 4
            target_model = copy.deepcopy(model)
            if self.args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=lr/2, momentum=0.5)
            elif self.args.optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=lr/2, weight_decay=5e-4)
        else:
            if self.args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.5)
            elif self.args.optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

        for iter in range(local_ep):
            batch_loss = []
            model.train()
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                new_images = images
                new_labels = labels
                if ((self.idx in adversary_list) and (self.global_round in poison_epoch[adversarial_index])
                        and not generate_trigger):
                    poison_count = min(poisoning_per_batch, images.size(0))
                    new_labels = labels.clone()
                    new_labels[:poison_count] = poison_label_swap
                    new_images = add_trigger_batch(
                        images,
                        adversarial_index,
                        dataset=self.args.dataset,
                        trigger_value=self.trigger_value,
                        limit=poison_count,
                    )
                images = new_images
                labels = new_labels
                images.requires_grad_(False)

                optimizer.zero_grad(set_to_none=True)
                log_probs = model(images)

                if (((self.idx in adversary_list) and (self.global_round in poison_epoch[adversarial_index])
                        and not generate_trigger)):
                    hooks = []
                    for name, module in model.named_modules():
                        hook = module.register_backward_hook(selective_grad_update_hook)
                        hooks.append(hook)
                    loss = 0.7 * self.criterion(log_probs, labels) + 0.3 * models.model_norm(model, target_model)
                    loss.backward()
                    optimizer.step()
                    for hook in hooks:
                        hook.remove()
                else:
                    loss = self.criterion(log_probs, labels)
                    loss.backward()
                    optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        self.global_round, iter, batch_idx * len(images), len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # if ((self.idx in adversary_list) and (self.global_round in poison_epoch[adversarial_index])
        #         and not generate_trigger):
        #     for key, value in model.state_dict().items():
        #         target_value = target_model.state_dict()[key]
        #         new_value = target_value + (value - target_value) * scale_weights
        #         model.state_dict()[key].copy_(new_value)

        # project_within_radius(model, target_model, 1.0)

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        model.eval()
        loss, total, correct, att_success = 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.testloader):
                adversarial_index = -1
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                new_images = images
                new_labels = labels

                if self.idx in adversary_list:
                    for temp_index in range(0, len(adversary_list)):
                        if int(self.idx) == adversary_list[temp_index]:
                            adversarial_index = temp_index % 3
                            break

                if (self.idx in adversary_list) and (self.global_round in poison_epoch[adversarial_index]):
                    new_labels = labels.clone()
                    new_labels.fill_(poison_label_swap)
                    new_images = add_trigger_batch(
                        images,
                        adversarial_index,
                        dataset=self.args.dataset,
                        trigger_value=self.trigger_value,
                    )
                images = new_images
                labels = new_labels

                outputs = model(images)
                batch_loss = self.criterion(outputs, labels)
                loss += batch_loss.item()

                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

        accuracy = correct / total
        return accuracy, loss


def test_inference(args, model, test_dataset, trigger_value, poisoned=True):

    model.eval()
    loss, total, correct, att_success = 0.0, 0.0, 0.0, 0.0

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    criterion = nn.NLLLoss().to(device)

    if poisoned:
        range_id = getattr(test_dataset, '_non_target_indices_cache', None)
        if range_id is None:
            range_id = []
            for ind, x in enumerate(test_dataset):
                _, label = x
                if label == poison_label_swap:
                    continue
                range_id.append(ind)
            test_dataset._non_target_indices_cache = range_id
        testloader = DataLoader(
            test_dataset,
            batch_size=512,
            shuffle=False,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(range_id),
        )
    else:
        testloader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            adversarial_index = -1
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            new_images = images
            new_labels = labels

            if poisoned:
                new_labels = labels.clone()
                new_labels.fill_(poison_label_swap)
                new_images = add_trigger_batch(
                    images,
                    adversarial_index,
                    dataset=args.dataset,
                    trigger_value=trigger_value,
                )
            images = new_images
            labels = new_labels

            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

    accuracy = correct / total
    return accuracy, loss
