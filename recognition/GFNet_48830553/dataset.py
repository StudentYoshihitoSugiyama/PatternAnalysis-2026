import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

DATA_TEST_PATH = "/home/groups/comp3710/ADNI/test/"
DATA_TRAIN_PATH = "/home/groups/comp3710/ADNI/train/"


class ADNIDataset(Dataset):
    """Custom Dataset for ADNI images."""

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
