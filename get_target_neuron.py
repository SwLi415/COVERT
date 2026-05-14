import math
import torch
import numpy as np


def get_top_k_neurons(weight_history, ratio):
    top_rank_neurons = {}  # 得分最高的神经元索引
    neuron_stats = {}  # 保存各层神经元统计信息

    for name, weights_list in weight_history.items():
        print(f"\n分析层: {name}")

        # 转为 NumPy 数组，形状为
        # [rounds, out_channels, in_channels, kernel_h, kernel_w]
        weights_array = np.array(weights_list)

        # 计算每轮每个神经元的平均绝对权重
        # shape: [rounds, out_channels]
        neuron_means_per_round = np.mean(np.abs(weights_array), axis=(2, 3, 4))

        # 计算相邻轮次之间的变化量
        # shape: [rounds - 1, out_channels]
        neuron_diffs = np.diff(neuron_means_per_round, axis=0)

        # 计算变化量的方差
        # shape: [out_channels]
        neuron_diff_vars = np.var(neuron_diffs, axis=0)

        # 1. 计算所有轮次中的平均权重
        neuron_means = np.mean(neuron_means_per_round, axis=0)

        # 2. 计算所有轮次中的最大波动幅度
        neuron_max_diff = np.max(neuron_means_per_round, axis=0) - np.min(neuron_means_per_round, axis=0)

        # 3. 取最后一轮的权重均值
        last_round_weights = neuron_means_per_round[-1]

        # 4. 综合评分:
        #    |最后一轮权重| * 变化量方差 / (最大波动 + epsilon)
        neuron_rank = (np.abs(last_round_weights) * neuron_diff_vars) / (neuron_max_diff + 1e-10)
        alpha = 1
        neuron_rank = pow(neuron_rank, alpha)

        neuron_stats[name] = {
            'means': neuron_means,
            'diff_vars': neuron_diff_vars,  # 变化量方差
            'max_diff': neuron_max_diff,
            'rank': neuron_rank,
            'weights_per_round': neuron_means_per_round,
            'diffs_per_round': neuron_diffs  # 每轮变化量
        }

        # 按比例选取 Top-K 神经元
        num_neurons = neuron_means.shape[0]
        top_k = max(1, int(ratio * num_neurons))
        top_rank_indices = np.argsort(neuron_rank)[-top_k:]
        top_rank_neurons[name] = top_rank_indices

    return top_rank_neurons
