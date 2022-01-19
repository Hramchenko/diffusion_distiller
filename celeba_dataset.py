from torch.utils.data import Dataset
from tensorfn.data import LMDBReader
from torchvision import transforms
import torch
import cv2
import numpy as np

class CelebaDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.reader = LMDBReader(path, reader="raw")

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, index):
        img_bytes = self.reader.get(
            f"{self.resolution}-{str(index).zfill(5)}".encode("utf-8")
        )

        data = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(data, 1)
        img = self.transform(img)

        return img

class CelebaWrapper(torch.utils.data.Dataset):

    def __init__(self, dataset_dir, resolution):
        super().__init__()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomHorizontalFlip()
        ])
        self.dataset = CelebaDataset(dataset_dir, transform=transform, resolution=resolution)

    def __getitem__(self, item):
        return self.dataset[item], 0

    def __len__(self):
        return len(self.dataset)
