# COVERT

The main entry point of this project is `federated_main.py`. If your primary goal is to run the code, you mainly need to focus on environment setup, data preparation, run commands, and output results.

## 1. Environment Setup

It is recommended to use a Python 3.9 environment or a similar version.

Common dependencies:

```bash
pip install torch torchvision tensorboardX tqdm timm pillow matplotlib numpy
```

If you use a GPU, please install the corresponding version of PyTorch according to your CUDA version.

## 2. Main Program Entry

The main execution script is:

```bash
python federated_main.py
```

The program will automatically complete:

- Parse command-line arguments
- Load the dataset and split client data
- Build the global model
- Perform federated training
- Evaluate test accuracy and attack success rate during training
- Save models, logs, and plots

## 3. Data Preparation

The current code supports the following datasets:

- `cifar`
- `tinyimagenet`
- `gtsrb`

The data paths are defined in `utils.py`, with the following defaults.

### CIFAR-10

Default path:

```text
./data/cifar/
```

Description:

- Uses `torchvision.datasets.CIFAR10`
- If it does not exist locally, it will be downloaded automatically on the first run

### Tiny-ImageNet

Default path:

```text
./data/tiny-imagenet-200/
```

The directory structure should be similar to:

```text
data/tiny-imagenet-200/
├─ train/
├─ val/
├─ wnids.txt
└─ words.txt
```

### GTSRB

Default path:

```text
./data/GTSRB/Final_Training/Images
```

The category images are expected to be organized into subfolders.

## 4. Pretrained Models

In the current `federated_main.py`, the pretrained model loading statements in the `cifar`, `tinyimagenet`, and `gtsrb` branches have all been commented out. That means:

- By default, the code starts training from random initialization
- It is no longer mandatory to prepare model files under `./pretrained/...` before running

If you want to re-enable pretrained models later, you can manually uncomment the corresponding `load_state_dict(...)` lines in `federated_main.py` and prepare the corresponding weight files.

## 5. Run Examples

### Example 1: CIFAR-10, IID

```bash
python federated_main.py --dataset cifar --iid 1 --epochs 300 --num_users 100 --frac 0.1 --local_ep 2 --local_bs 32 --optimizer adam --lr 0.001
```

### Example 2: CIFAR-10, Non-IID

```bash
python federated_main.py --dataset cifar --iid 0 --epochs 300 --num_users 100 --frac 0.1 --local_ep 2 --local_bs 32 --optimizer adam --lr 0.001
```

## 6. Common Parameter Descriptions

The main parameters are defined in `options.py`.

- `--dataset`: Dataset name, options are `cifar`, `tinyimagenet`, `gtsrb`
- `--epochs`: Number of federated training rounds
- `--num_users`: Total number of clients
- `--frac`: Proportion of clients participating in each round
- `--local_ep`: Number of local training epochs for each client
- `--local_bs`: Local batch size
- `--lr`: Learning rate of the local optimizer
- `--global_lr`: Global aggregation learning rate, default is `1.0`
- `--optimizer`: Optimizer type, options are `sgd` or `adam`
- `--iid`: Whether to use IID partitioning, `1` means IID and `0` means Non-IID
- `--verbose`: Whether to output more detailed training logs

## 7. Output Results

After the current code runs, common outputs include:

- Training logs in the console
- `output.txt`: Standard output redirection log
- `logs/`: TensorBoard logs
- `save/global_model.pth`: Global model after training
- `save/objects/*.pkl`: Records of training loss and training accuracy
- `save/*.png`: Curves of loss, accuracy, and attack success rate
