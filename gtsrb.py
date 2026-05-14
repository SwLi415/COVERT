import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class GTSRBDataset(Dataset):
    def __init__(self, root_dir, train=True, split_ratio=0.8, transform=None, seed=42):
        """
        Args:
            root_dir (str): GTSRB 数据集的根目录，应该是 `GTSRB/Final_Training/Images/`。
            train (bool): 是否加载训练集 (True) 或测试集 (False)。
            split_ratio (float): 训练集和测试集的划分比例 (默认为 0.8)。
            transform (callable, optional): 数据增强转换。
            seed (int): 用于划分数据集的随机种子，确保划分一致性。
        """
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.split_ratio = split_ratio
        self.seed = seed

        # 加载图像和标签
        self.data = self._load_data()
        self.train_data, self.test_data = self._split_data()

    def _load_data(self):
        """
        遍历主目录，加载所有图像路径和对应的标签。
        """
        data = []
        for label_folder in os.listdir(self.root_dir):
            label_folder_path = os.path.join(self.root_dir, label_folder)
            if os.path.isdir(label_folder_path):
                # 类别标签为文件夹名（需转换为整数）
                label = int(label_folder)
                for file_name in os.listdir(label_folder_path):
                    if file_name.endswith(('.ppm', '.jpg', '.png')):  # 支持的图像格式
                        file_path = os.path.join(label_folder_path, file_name)
                        data.append((file_path, label))
        return data

    def _split_data(self):
        """
        根据 split_ratio 划分数据集为训练集和测试集。
        """
        random.seed(self.seed)  # 设置随机种子，确保划分一致性
        random.shuffle(self.data)

        split_idx = int(len(self.data) * self.split_ratio)
        train_data = self.data[:split_idx]
        test_data = self.data[split_idx:]
        return train_data, test_data

    def __len__(self):
        """
        返回训练集或测试集的大小。
        """
        return len(self.train_data) if self.train else len(self.test_data)

    def __getitem__(self, idx):
        """
        返回图像和标签。
        """
        data = self.train_data if self.train else self.test_data
        img_path, label = data[idx]

        # 加载图像
        image = Image.open(img_path).convert('RGB')  # 确保为 RGB 格式
        if self.transform:
            image = self.transform(image)

        return image, label

# 使用示例
if __name__ == "__main__":
    # 定义数据增强
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # 缩放到 32x32
        transforms.ToTensor(),       # 转换为张量
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
    ])

    # 数据集路径（解压后的路径）
    root_dir = "/path/to/GTSRB/Final_Training/Images"

    # 加载训练集
    train_dataset = GTSRBDataset(root_dir=root_dir, train=True, transform=transform)

    # 加载测试集
    test_dataset = GTSRBDataset(root_dir=root_dir, train=False, transform=transform)

    # 测试加载数据
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
