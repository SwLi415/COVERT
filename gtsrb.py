import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class GTSRBDataset(Dataset):
    def __init__(self, root_dir, train=True, split_ratio=0.8, transform=None, seed=42):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.split_ratio = split_ratio
        self.seed = seed

        self.data = self._load_data()
        self.train_data, self.test_data = self._split_data()

    def _load_data(self):
        data = []
        for label_folder in os.listdir(self.root_dir):
            label_folder_path = os.path.join(self.root_dir, label_folder)
            if os.path.isdir(label_folder_path):
                label = int(label_folder)
                for file_name in os.listdir(label_folder_path):
                    if file_name.endswith(('.ppm', '.jpg', '.png')):
                        file_path = os.path.join(label_folder_path, file_name)
                        data.append((file_path, label))
        return data

    def _split_data(self):
        random.seed(self.seed)
        random.shuffle(self.data)

        split_idx = int(len(self.data) * self.split_ratio)
        train_data = self.data[:split_idx]
        test_data = self.data[split_idx:]
        return train_data, test_data

    def __len__(self):
        return len(self.train_data) if self.train else len(self.test_data)

    def __getitem__(self, idx):
        data = self.train_data if self.train else self.test_data
        img_path, label = data[idx]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    root_dir = "/path/to/GTSRB/Final_Training/Images"

    train_dataset = GTSRBDataset(root_dir=root_dir, train=True, transform=transform)

    test_dataset = GTSRBDataset(root_dir=root_dir, train=False, transform=transform)

    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
