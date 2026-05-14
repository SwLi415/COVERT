import torch
import os
import numpy as np
import hdbscan
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_context('poster')
sns.set_style('white')
sns.set_color_codes()
plot_kwds = {'alpha': 0.5, 's': 80, 'linewidths': 0}


def hdbscan_clients(similarities):
    hdbscan_clustering = hdbscan.HDBSCAN(min_cluster_size=10, gen_min_span_tree=True)
    cluster_result = hdbscan_clustering.fit(similarities)
    return cluster_result


def compute_pairwise_similarity(local_param, global_param, i):
    if local_param is not None and global_param is not None:
        score = torch.cosine_similarity(
            flatten_and_concat(global_param),
            flatten_and_concat(local_param),
            dim=0,
            eps=1e-12,
        ).item()
        similarity_matrix[i, 0] = score


def flatten_and_concat(src):
    return torch.cat([tensor.flatten() for tensor in src if tensor is not None])


device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    similarity_matrix = np.zeros((100, 1))
    path = './save/local_model/'
    models = os.listdir(path)

    global_model_param = torch.load('./save/global_model.pth')

    for i, model in enumerate(models):
        local_model_param = torch.load(path + model)
        compute_pairwise_similarity(flatten_and_concat(local_model_param.values()),
                                    flatten_and_concat(global_model_param.values()), i)
    print(similarity_matrix)

    cluster_result = hdbscan_clients(similarity_matrix)
    print('cluster label: ', cluster_result.labels_)
    print('class probability: ', cluster_result.probabilities_)

    # 可视化
    palette = sns.color_palette()
    cluster_colors = [sns.desaturate(palette[col], sat)
                      if col >= 0 else (0.5, 0.5, 0.5) for col, sat in
                      zip(cluster_result.labels_, cluster_result.probabilities_)]
    y = [0 for _ in range(similarity_matrix.shape[0])]
    plt.scatter(similarity_matrix.T[0], y, c=cluster_colors, **plot_kwds)
    plt.savefig('./save/cluster_result.png')
