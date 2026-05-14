import torch
import copy
import numpy as np
from hdbscan import HDBSCAN


def _flatten_model(model_update):
    k_list = list(model_update.keys())
    return torch.cat([model_update[k].flatten() for k in k_list])


def _recon_model(template, flatten_update):
    cpy_update = copy.deepcopy(template)
    start_ = 0
    for k in cpy_update.keys():
        v = cpy_update[k]
        numel = v.numel()
        v.copy_(flatten_update[start_:start_ + numel].reshape(v.shape))
        start_ += numel
    return cpy_update


def _cluster(flatten_updates):
    flatten_updates_cpu = [update.cpu().numpy() for update in flatten_updates]
    min_cluster_size = max(2, int(np.ceil(len(flatten_updates_cpu)/1.5)))
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, allow_single_cluster=True)
    cluster_labels = clusterer.fit_predict(flatten_updates_cpu)
    benign_idx = np.argwhere(cluster_labels == 0).flatten()
    return benign_idx


def _adaptive_clip(flatten_update, standard_norm):
    return flatten_update * min(1.0, standard_norm / torch.norm(flatten_update, p=2))


def _add_dp_noise_(model, standard_norm, lam=0.000001):
    for key in model.keys():
        param = model[key]
        noise = torch.normal(mean=0, std=lam, size=param.shape, device=param.device)
        param.add_(noise)


def flame(w):
    """
    FLAME-based federated aggregation.
    Input/output format is the same as average_weights().
    """
    flatten_updates = [_flatten_model(model_update) for model_update in w]
    benign_idx = _cluster(flatten_updates)
    print(benign_idx)
    flatten_updates = [flatten_updates[i] for i in benign_idx]
    if len(flatten_updates) == 0:  # fallback, all are outliers
        flatten_updates = [_flatten_model(model_update) for model_update in w]

    median_norm = np.median([torch.norm(update, p=2).item() for update in flatten_updates])
    template = copy.deepcopy(w[0])
    cliped_updates = [_recon_model(template, _adaptive_clip(update, median_norm)) for update in flatten_updates]

    # 平均聚合
    w_avg = copy.deepcopy(cliped_updates[0])
    for key in w_avg.keys():
        for i in range(1, len(cliped_updates)):
            w_avg[key] += cliped_updates[i][key]
        w_avg[key] = torch.div(w_avg[key], len(cliped_updates))

    # _add_dp_noise_(w_avg, median_norm)

    return w_avg
