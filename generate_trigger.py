import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image


def optimize_backdoor_trigger_weights_based(model, layer_neuron_indices, target_class,
                                            input_shape=(3, 32, 32), trigger_positions=None,
                                            train_data_loader=None, num_train_samples=20,
                                            num_iterations=100, trigger_lr=0.1, train_lr=0.1,
                                            train_steps=10, init_pattern=None):
    """
    优化触发器，使带触发器的样本在模拟训练时更容易影响目标神经元权重。
    当前实现只关注权重增大这一目标，不额外加入其它正则项。

    参数:
        model: 基础模型
        layer_neuron_indices: 目标神经元信息，格式如 {'layer_name.weight': [neuron_indices]}
        target_class: 后门目标类别
        input_shape: 输入图像形状
        trigger_positions: 触发器固定位置
        train_data_loader: 训练数据加载器
        num_train_samples: 每次模拟训练使用的样本数
        num_iterations: 触发器优化迭代次数
        trigger_lr: 触发器学习率
        train_lr: 模拟训练学习率
        train_steps: 每次迭代中的模拟训练步数
        init_pattern: 初始触发器图案

    返回:
        best_trigger: 优化过程中得到的最佳触发器
        trigger_mask: 触发器掩码
        optimization_history: 优化历史记录
    """
    import copy
    device = next(model.parameters()).device

    # 初始化触发器
    if init_pattern is None:
        pattern = torch.rand(input_shape, device=device) * 0.1
    else:
        pattern = init_pattern.clone().to(device)

    # 创建可优化触发器
    trigger = torch.nn.Parameter(pattern.clone(), requires_grad=True)

    # 创建固定位置的触发器掩码
    mask = torch.zeros(input_shape, device=device)
    for position in trigger_positions:
        if position[0] < input_shape[0] and position[1] < input_shape[1] and position[2] < input_shape[2]:
            mask[position[0], position[1], position[2]] = 1.0
        else:
            print(f"警告: 位置 [{position[0]},{position[1]},{position[2]}] 超出图像范围 {input_shape}")

    # 准备用于模拟训练的子集数据
    train_images = []
    train_labels = []

    if train_data_loader is not None:
        # 收集一部分训练样本
        collected_samples = 0
        for images, labels in train_data_loader:
            batch_size = images.size(0)
            if collected_samples + batch_size > num_train_samples:
                take_samples = num_train_samples - collected_samples
                train_images.append(images[:take_samples])
                train_labels.append(labels[:take_samples])
                collected_samples += take_samples
                break
            else:
                train_images.append(images)
                train_labels.append(labels)
                collected_samples += batch_size

            if collected_samples >= num_train_samples:
                break

        train_images = torch.cat(train_images, dim=0).to(device)
        train_labels = torch.cat(train_labels, dim=0).to(device)

        print(f"收集了 {len(train_images)} 个训练样本用于模拟训练")
    else:
        print("警告: 未提供训练数据，将使用随机生成的数据")
        train_images = torch.rand(num_train_samples, *input_shape, device=device)
        train_labels = torch.randint(0, 10, (num_train_samples,), device=device)

    # 创建触发器优化器
    trigger_optimizer = torch.optim.Adam([trigger], lr=trigger_lr)

    # 建立层名到模块的映射
    layer_module_map = {}
    for weight_name, indices in layer_neuron_indices.items():
        layer_name = weight_name.split('.weight')[0]
        for name, module in model.named_modules():
            if name == layer_name and hasattr(module, 'weight'):
                layer_module_map[weight_name] = (name, module)
                break

    # 记录优化历史
    history = {
        'weight_changes': [],
        'loss': [],
        'neuron_weights': {
            weight_name: {
                neuron_idx: [] for neuron_idx in indices
            } for weight_name, indices in layer_neuron_indices.items()
        }
    }

    criterion = torch.nn.CrossEntropyLoss()

    # 保存当前最优触发器
    best_loss = float('inf')
    best_trigger = None
    best_iteration = -1

    class SaveActivations(torch.nn.Module):
        """用于记录中间层激活的辅助模块。"""

        def __init__(self):
            super().__init__()
            self.activations = {}

        def reset(self):
            self.activations = {}

        def save_activation(self, name, module, input, output):
            self.activations[name] = output

    # 触发器优化循环
    for iteration in range(num_iterations):
        trigger_optimizer.zero_grad()

        # 当前触发器模式
        triggered_pattern = (trigger * mask)

        # 将触发器添加到部分训练样本中
        num_poisoned = min(len(train_images) // 2, 32)  # 最多投毒 32 个样本
        poisoned_indices = torch.randperm(len(train_images))[:num_poisoned]
        clean_indices = torch.randperm(len(train_images))[num_poisoned:num_poisoned * 2]

        poisoned_images = train_images[poisoned_indices].clone()
        poisoned_labels = torch.full((num_poisoned,), target_class, device=device)

        poisoned_images = poisoned_images * (1 - mask) + triggered_pattern.unsqueeze(0) * mask
        poisoned_images = torch.clamp(poisoned_images, 0, 1)

        # 构造模拟训练数据
        sim_images = torch.cat([poisoned_images, train_images[clean_indices]], dim=0)
        sim_labels = torch.cat([poisoned_labels, train_labels[clean_indices]], dim=0)

        # 创建模型副本进行模拟训练
        sim_model = copy.deepcopy(model)
        sim_model.train()

        # 记录训练前目标神经元权重
        initial_weights = {}
        for weight_name, (name, module) in layer_module_map.items():
            initial_weights[weight_name] = {}
            for neuron_idx in layer_neuron_indices[weight_name]:
                if neuron_idx < module.weight.shape[0]:
                    initial_weights[weight_name][neuron_idx] = module.weight[neuron_idx].clone().detach()

                    weight_mean = torch.mean(torch.abs(module.weight[neuron_idx])).item()
                    history['neuron_weights'][weight_name][neuron_idx].append(weight_mean)

        # 模拟训练
        sim_optimizer = torch.optim.SGD(sim_model.parameters(), lr=train_lr)

        for step in range(train_steps):
            perm = torch.randperm(len(sim_images), device=device)
            select_indices = perm[:4]
            sim_batch_images = sim_images[select_indices]
            sim_batch_labels = sim_labels[select_indices]

            sim_optimizer.zero_grad()
            outputs = sim_model(sim_batch_images)
            loss = criterion(outputs, sim_batch_labels)
            loss.backward(retain_graph=True)
            sim_optimizer.step()

        # 计算目标神经元权重变化
        weight_increase_loss = 0
        weight_changes = []

        # 再次前向传播，建立触发器相关的计算图
        mini_batch_size = min(5, num_poisoned)
        mini_batch_images = train_images[:mini_batch_size].clone()
        mini_batch_images = mini_batch_images * (1 - mask) + triggered_pattern.unsqueeze(0) * mask
        mini_batch_images = torch.clamp(mini_batch_images, 0, 1)
        mini_batch_labels = torch.full((mini_batch_size,), target_class, device=device)

        outputs = sim_model(mini_batch_images)
        classification_loss = criterion(outputs, mini_batch_labels)

        for weight_name, (name, original_module) in layer_module_map.items():
            for layer_name, module in sim_model.named_modules():
                if layer_name == name:
                    for neuron_idx in layer_neuron_indices[weight_name]:
                        if neuron_idx < module.weight.shape[0] and neuron_idx in initial_weights[weight_name]:
                            initial_weight = initial_weights[weight_name][neuron_idx]
                            current_weight = module.weight[neuron_idx]

                            weight_diff = current_weight - initial_weight
                            weight_change = torch.mean(weight_diff).item()
                            weight_changes.append(weight_change)

                            # 希望目标神经元权重增大，因此使用负号
                            weight_increase_loss -= torch.mean(module.weight[neuron_idx]) * 0.1

                            # 增加一条与输出相关的弱连接，帮助梯度传回触发器
                            extra_loss = torch.mean(torch.abs(outputs)) * 0.000001
                            weight_increase_loss += extra_loss

        avg_weight_change = sum(weight_changes) / len(weight_changes) if weight_changes else 0
        history['weight_changes'].append(avg_weight_change)

        total_loss = weight_increase_loss

        try:
            if not total_loss.requires_grad:
                print(f"警告: 第 {iteration + 1} 次迭代的损失没有梯度")
                total_loss = weight_increase_loss + torch.mean(trigger) * 0.0001

            history['loss'].append(total_loss.item())

            current_loss = total_loss.item()
            if current_loss < best_loss:
                best_loss = current_loss
                best_trigger = trigger.clone().detach().cpu()
                best_iteration = iteration
                print(f"在迭代 {iteration + 1} 时发现更好的触发器，损失 {best_loss:.4f}")

            total_loss.backward()
            trigger_optimizer.step()

        except RuntimeError as e:
            print(f"迭代 {iteration + 1} 出错: {e}")
            history['loss'].append(float('nan'))

            fallback_loss = torch.mean(trigger * 0.1)
            fallback_loss.backward()
            trigger_optimizer.step()

        if (iteration + 1) % 10 == 0:
            print(f"迭代 {iteration + 1}/{num_iterations}:")
            print(f"  权重增大损失: {weight_increase_loss.item():.4f}")
            print(f"  平均权重变化: {avg_weight_change:.6f}")
            print(f"  当前最优: {best_loss:.4f} (迭代 {best_iteration + 1})")

    print(f"优化完成。最佳触发器来自迭代 {best_iteration + 1}/{num_iterations}，损失 {best_loss:.4f}")

    if best_trigger is None:
        best_trigger = trigger.detach().cpu()

    trigger_mask = mask.cpu()

    return best_trigger, trigger_mask, history
