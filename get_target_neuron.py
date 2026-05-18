import math
import torch
import numpy as np


def get_top_k_neurons(weight_history, ratio):
    top_rank_neurons = {}
    neuron_stats = {}

    for name, weights_list in weight_history.items():
        print(f"\n分析层: {name}")

        weights_array = np.array(weights_list)

        neuron_means_per_round = np.mean(np.abs(weights_array), axis=(2, 3, 4))

        neuron_diffs = np.diff(neuron_means_per_round, axis=0)

        neuron_diff_vars = np.var(neuron_diffs, axis=0)

        neuron_means = np.mean(neuron_means_per_round, axis=0)

        neuron_max_diff = np.max(neuron_means_per_round, axis=0) - np.min(neuron_means_per_round, axis=0)

        last_round_weights = neuron_means_per_round[-1]

        neuron_rank = (np.abs(last_round_weights) * neuron_diff_vars) / (neuron_max_diff + 1e-10)
        alpha = 1
        neuron_rank = pow(neuron_rank, alpha)

        neuron_stats[name] = {
            'means': neuron_means,
            'diff_vars': neuron_diff_vars,
            'max_diff': neuron_max_diff,
            'rank': neuron_rank,
            'weights_per_round': neuron_means_per_round,
            'diffs_per_round': neuron_diffs
        }

        num_neurons = neuron_means.shape[0]
        top_k = max(1, int(ratio * num_neurons))
        top_rank_indices = np.argsort(neuron_rank)[-top_k:]
        top_rank_neurons[name] = top_rank_indices

    return top_rank_neurons
