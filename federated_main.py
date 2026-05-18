import os
import sys
import time
import pickle
import copy
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Subset
from options import args_parser
from collections import defaultdict
from PIL import Image
from update import LocalUpdate, test_inference, poison_epoch
from utils import get_dataset, average_weights, exp_details
from resnet import ResNet18
from get_target_neuron import get_top_k_neurons
from generate_trigger import optimize_backdoor_trigger_weights_based
from astroformer import astroformer_3


device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
layer_neuron_indices = None
bottom = None
trigger_value = torch.rand((3, 32, 32), device=device) * 0.1
trigger_positions = [
    [0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3],
    [1, 1, 4], [1, 1, 5], [1, 1, 6], [1, 1, 7],
    [2, 2, 6], [2, 2, 7], [2, 2, 8], [2, 2, 9]
]


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


type = sys.getfilesystemencoding()
sys.stdout = Logger("output.txt")


def flatten_and_concat(src):
    return torch.cat([tensor.flatten() for tensor in src if tensor is not None])


def build_local_update(args, train_dataset, user_groups, logger, idx, epoch, layer_neuron_indices, trigger_value):
    return LocalUpdate(
        args=args,
        dataset=train_dataset,
        idxs=user_groups[idx],
        logger=logger,
        idx=idx,
        layer_neuron_indices=layer_neuron_indices,
        global_round=epoch,
        trigger_value=trigger_value,
    )


if __name__ == '__main__':
    start_time = time.time()

    path_project = os.path.abspath('..')
    logger = SummaryWriter('./logs')

    args = args_parser()
    exp_details(args)

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    train_dataset, test_dataset, user_groups = get_dataset(args)
    trigger_train_loader = DataLoader(train_dataset, batch_size=args.local_bs, shuffle=True)

    if args.dataset == 'cifar':
        global_model = ResNet18(10)
        # global_model.load_state_dict(torch.load('./pretrained/cifar/global_model_cifar.pth'))
    elif args.dataset == 'tinyimagenet':
        global_model = astroformer_3(
            pretrained=False,
            in_chans=3,
            num_classes=200,
            drop_rate=0.1,
        )
        # global_model.load_state_dict(torch.load('./pretrained/imagenet/global_model_imagenet.pth', map_location=device))
    elif args.dataset == 'gtsrb':
        global_model = ResNet18(43)
        # global_model.load_state_dict(torch.load('./pretrained/gtsrb/global_model_gtsrb.pth', map_location=device))

    global_model.to(device)
    global_model.train()
    print(global_model)

    global_weights = global_model.state_dict()

    train_loss, train_accuracy = [], []
    test_accuracy = []
    attack_success_rate = []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0
    weight_history = {}
    conv_layers = {}
    for name, param in global_model.named_parameters():
        if 'weight' in name and len(param.shape) == 4:
            conv_layers[name] = True

    print(f"找到以下卷积层: {list(conv_layers.keys())}")

    benign_label = [True] * args.num_users
    print('初始化本地模型动态标签:', benign_label)

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # if epoch in poison_epoch[0] and 0 not in idxs_users:
        #     idxs_users[0] = 0
        # elif epoch in poison_epoch[1] and 1 not in idxs_users:
        #     idxs_users[0] = 1
        # elif epoch in poison_epoch[2] and 2 not in idxs_users:
        #     idxs_users[0] = 2

        if 11 < epoch < 112:
            for i in range(3):
                if i not in idxs_users:
                    for j in range(len(idxs_users)):
                        if idxs_users[j] not in [0, 1, 2]:
                            idxs_users[j] = i
                            break
        print('Randomly select the local model id to be: ', idxs_users)

        for idx in idxs_users:
            print(idx)
            local_model = build_local_update(
                args, train_dataset, user_groups, logger, idx, epoch, layer_neuron_indices, trigger_value
            )
            w, loss = local_model.update_weights(model=copy.deepcopy(global_model))
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        global_lr = args.global_lr
        aggregated_param = average_weights(local_weights)
        for key, value in global_model.state_dict().items():
            new_value = value + (aggregated_param[key] - value) * global_lr
            global_model.state_dict()[key].copy_(new_value)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        global_model.eval()
        list_acc, list_loss = [], []
        for c in range(args.num_users):
            local_model = build_local_update(
                args, train_dataset, user_groups, logger, c, epoch, layer_neuron_indices, trigger_value
            )
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc) / len(list_acc))

        print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
        print(f'Training Loss : {train_loss[-1]}')
        print('Train Accuracy: {:.2f}% '.format(100 * train_accuracy[-1]))
        asr, _ = test_inference(args, global_model, test_dataset, trigger_value, poisoned=True)
        print('Attack Success Rate: {:.2f}%\n'.format(100 * asr))
        attack_success_rate.append(asr)
        acc, _ = test_inference(args, global_model, test_dataset, trigger_value, poisoned=False)
        print('Test Accuracy: {:.2f}%\n'.format(100 * acc))
        test_accuracy.append(acc)

        if epoch in range(0, 10):
            for name in conv_layers:
                weight_data = global_model.state_dict()[name].clone().cpu().numpy()
                if name not in weight_history:
                    weight_history[name] = []
                weight_history[name].append(weight_data)
        if epoch == 10:
            layer_neuron_indices = get_top_k_neurons(weight_history, ratio=0.3)
            trigger_value, _, _ = optimize_backdoor_trigger_weights_based(
                global_model,
                layer_neuron_indices,
                2,
                input_shape=(3, 32, 32),
                trigger_positions=trigger_positions,
                train_data_loader=trigger_train_loader,
                num_iterations=500,
                trigger_lr=0.2,
            )

    test_acc, test_loss = test_inference(args, global_model, test_dataset, trigger_value, poisoned=False)
    test_asr, _ = test_inference(args, global_model, test_dataset, trigger_value, poisoned=True)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))
    print("|---- Attack Success Rate: {:.2f}%".format(100 * test_asr))

    torch.save(global_model.state_dict(), './save/global_model.pth')

    file_name = './save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'. \
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use('Agg')

    plt.figure(figsize=(12, 9))
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs))

    plt.figure(figsize=(12, 9))
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs))

    plt.figure(figsize=(12, 9))
    plt.title('Attack Success Rate vs Communication rounds')
    plt.plot(range(len(attack_success_rate)), attack_success_rate, color='r')
    plt.ylabel('Attack Success Rate')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_asr.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs))

    plt.figure(figsize=(12, 9))
    plt.title('Test Accuracy vs Communication rounds')
    plt.plot(range(len(test_accuracy)), test_accuracy, color='r')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_main_acc.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs))
