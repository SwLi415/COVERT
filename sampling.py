import numpy as np
from torchvision import datasets, transforms


def _iid_partition_indices(dataset_len, num_users, divisor=1):
    num_items = int(dataset_len / (divisor * num_users))
    shuffled_indices = np.random.permutation(dataset_len)
    dict_users = {}
    for i in range(num_users):
        start = i * num_items
        end = start + num_items
        dict_users[i] = set(shuffled_indices[start:end].tolist())
    return dict_users


def _dirichlet_noniid_partition(labels, num_users, alpha=0.5):
    labels = np.array(labels)
    n_samples = len(labels)
    n_classes = len(np.unique(labels))

    avg_size = n_samples // num_users
    min_size = max(10, int(avg_size * 0.4))

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    all_indices = np.random.permutation(n_samples)

    base_total = min_size * num_users
    if base_total >= n_samples:
        min_size = max(1, n_samples // (2 * num_users))
        base_total = min_size * num_users

    base_indices = all_indices[:base_total]
    remain_indices = all_indices[base_total:]

    for i in range(num_users):
        dict_users[i] = base_indices[i * min_size:(i + 1) * min_size]

    remain_labels = labels[remain_indices]

    for k in range(n_classes):
        idx_k = remain_indices[remain_labels == k]
        np.random.shuffle(idx_k)

        if len(idx_k) == 0:
            continue

        proportions = np.random.dirichlet(np.repeat(alpha, num_users))
        split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        split_data = np.split(idx_k, split_points)

        for i in range(num_users):
            if len(split_data[i]) > 0:
                dict_users[i] = np.concatenate((dict_users[i], split_data[i]), axis=0)

    for i in range(num_users):
        np.random.shuffle(dict_users[i])

    return dict_users


def _get_tiny_imagenet_labels(dataset):
    return [label for _, label in dataset.images]


def _get_gtsrb_labels(dataset):
    data = dataset.train_data if dataset.train else dataset.test_data
    return [label for _, label in data]


def cifar_iid(dataset, num_users):
    return _iid_partition_indices(len(dataset), num_users)


def cifar_noniid(dataset, num_users, alpha=0.5):
    return _dirichlet_noniid_partition(dataset.targets, num_users, alpha)


def tiny_imagenet_iid(dataset, num_users):
    return _iid_partition_indices(len(dataset), num_users, divisor=2)


def tiny_imagenet_noniid(dataset, num_users, alpha=0.5):
    return _dirichlet_noniid_partition(_get_tiny_imagenet_labels(dataset), num_users, alpha)


def gtsrb_iid(dataset, num_users):
    return _iid_partition_indices(len(dataset), num_users)


def gtsrb_noniid(dataset, num_users, alpha=0.5):
    return _dirichlet_noniid_partition(_get_gtsrb_labels(dataset), num_users, alpha)
