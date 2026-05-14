# COVERT

本项目的主运行入口为 `federated_main.py`。如果你主要是想运行代码，重点关注环境准备、数据准备、运行命令和输出结果即可。

## 1. 环境准备

建议使用 Python 3.9 左右的环境。

常用依赖：

```bash
pip install torch torchvision tensorboardX tqdm timm pillow matplotlib numpy
```

如果使用 GPU，请根据自己的 CUDA 版本安装对应版本的 PyTorch。

## 2. 主程序入口

主运行脚本为：

```bash
python federated_main.py
```

程序会自动完成：

- 解析命令行参数
- 加载数据集并划分客户端数据
- 构建全局模型
- 执行联邦训练
- 在训练过程中评估测试精度和攻击成功率
- 保存模型、日志和曲线图

## 3. 数据准备

当前代码支持以下数据集：

- `cifar`
- `tinyimagenet`
- `gtsrb`

数据路径定义在 `utils.py` 中，默认如下。

### CIFAR-10

默认路径：

```text
./data/cifar/
```

说明：

- 使用 `torchvision.datasets.CIFAR10`
- 若本地不存在，会在首次运行时自动下载

### Tiny-ImageNet

默认路径：

```text
./data/tiny-imagenet-200/
```

目录结构应类似：

```text
data/tiny-imagenet-200/
├─ train/
├─ val/
├─ wnids.txt
└─ words.txt
```

### GTSRB

默认路径：

```text
./data/GTSRB/Final_Training/Images
```

要求类别图片按子文件夹组织。

## 4. 预训练模型

当前 `federated_main.py` 中，`cifar`、`tinyimagenet` 和 `gtsrb` 分支里的预训练模型加载语句都已经被注释掉了。也就是说：

- 默认情况下，代码会从随机初始化开始训练
- 运行前不再强制要求准备 `./pretrained/...` 下的模型文件

如果你之后希望重新启用预训练模型，可以手动取消 `federated_main.py` 中对应的 `load_state_dict(...)` 注释，并准备相应权重文件。

## 5. 运行示例

### 示例 1：CIFAR-10，IID

```bash
python federated_main.py --dataset cifar --iid 1 --epochs 300 --num_users 100 --frac 0.1 --local_ep 2 --local_bs 32 --optimizer adam --lr 0.001
```

### 示例 2：CIFAR-10，Non-IID

```bash
python federated_main.py --dataset cifar --iid 0 --epochs 300 --num_users 100 --frac 0.1 --local_ep 2 --local_bs 32 --optimizer adam --lr 0.001
```

## 6. 常用参数说明

主要参数定义在 `options.py` 中。

- `--dataset`：数据集名称，可选 `cifar`、`tinyimagenet`、`gtsrb`
- `--epochs`：联邦训练轮数
- `--num_users`：客户端总数
- `--frac`：每轮参与训练的客户端比例
- `--local_ep`：每个客户端的本地训练轮数
- `--local_bs`：本地 batch size
- `--lr`：本地优化器学习率
- `--global_lr`：全局聚合学习率，默认 `1.0`
- `--optimizer`：优化器类型，可选 `sgd` 或 `adam`
- `--iid`：是否采用 IID 划分，`1` 表示 IID，`0` 表示 Non-IID
- `--verbose`：是否输出更详细的训练日志

## 7. 输出结果

当前代码运行后，常见输出包括：

- 控制台训练日志
- `output.txt`：标准输出重定向日志
- `logs/`：TensorBoard 日志
- `save/global_model.pth`：训练完成后的全局模型
- `save/objects/*.pkl`：训练损失和训练精度记录
- `save/*.png`：损失、精度、攻击成功率曲线图
