import copy
import torch
import numpy as np
from torchvision import datasets, transforms
from sampling import (
    cifar_iid,
    cifar_noniid,
    tiny_imagenet_iid,
    tiny_imagenet_noniid,
    gtsrb_iid,
    gtsrb_noniid,
)
from tiny_imagenet import TinyImageNet
from gtsrb import GTSRBDataset
from torch.nn.utils import parameters_to_vector


def get_dataset(args):

    if args.dataset == 'cifar':
        data_dir = './data/cifar/'
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=train_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=test_transform)

        if args.iid:
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            if args.unequal:
                raise NotImplementedError()
            else:
                user_groups = cifar_noniid(train_dataset, args.num_users, 0.9)
    elif args.dataset == 'tinyimagenet':
        data_dir = './data/tiny-imagenet-200/'
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_dataset = TinyImageNet(data_dir, train=True, transform=train_transform)
        test_dataset = TinyImageNet(data_dir, train=False, transform=test_transform)
        if args.iid:
            user_groups = tiny_imagenet_iid(train_dataset, args.num_users)
        else:
            user_groups = tiny_imagenet_noniid(train_dataset, args.num_users, 0.9)

    elif args.dataset == 'gtsrb':
        data_dir = './data/GTSRB/Final_Training/Images'
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        train_dataset = GTSRBDataset(data_dir, train=True, transform=train_transform)

        test_dataset = GTSRBDataset(data_dir, train=False, transform=test_transform)
        if args.iid:
            user_groups = gtsrb_iid(train_dataset, args.num_users)
        else:
            user_groups = gtsrb_noniid(train_dataset, args.num_users, 0.9)

    else:
        print('Unknown dataset!')
    return train_dataset, test_dataset, user_groups


def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return


def project_within_radius(model, global_model, r=1.0):
    model_vec = torch.cat([p.data.view(-1) for p in model.parameters()])
    global_vec = torch.cat([p.data.view(-1) for p in global_model.parameters()])
    diff = model_vec - global_vec
    norm = diff.norm()
    print(norm)
    if norm > r:
        diff = diff * (r / norm)
        new_params = global_vec + diff
        idx = 0
        for p in model.parameters():
            numel = p.data.numel()
            p.data.copy_(new_params[idx:idx+numel].view_as(p.data))
            idx += numel
