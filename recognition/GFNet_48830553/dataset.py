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


def data_preprocess():
    """
    Defines image transformations and normalisation pipelines.
    
    Returns:
        tuple: (train_transform, eval_transform)
    """
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, eval_transform


def data_split(train_dir=DATA_TRAIN_PATH, split_ratio=0.8, seed=42):
    """
    Splits images in the training directory into training and validation sets
    at the patient level to prevent data leakage.

    Label mapping:
        AD -> 1
        NC -> 0

    Args:
        train_dir (str): Root directory of training data containing 'AD' and 'NC'.
        split_ratio (float): Ratio of patients to use for training (default: 0.8).
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (train_samples, val_samples) where each is a list of (image_path, label).
    """
    train_samples = []
    val_samples = []

    classes = [("NC", 0), ("AD", 1)]

    for class_name, label in classes:
        class_dir = os.path.join(train_dir, class_name)
        if not os.path.exists(class_dir):
            continue

        patient_to_images = {}
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpeg", ".jpg")):
                # File naming convention: <patient_id>_<slice_number>.jpeg
                patient_id = fname.split("_")[0]
                img_path = os.path.join(class_dir, fname)
                patient_to_images.setdefault(patient_id, []).append((img_path, label))

        patient_ids = sorted(list(patient_to_images.keys()))
        rng = random.Random(seed)
        rng.shuffle(patient_ids)

        split_idx = int(len(patient_ids) * split_ratio)
        train_patients = set(patient_ids[:split_idx])
        val_patients = set(patient_ids[split_idx:])

        for pid in train_patients:
            train_samples.extend(patient_to_images[pid])

        for pid in val_patients:
            val_samples.extend(patient_to_images[pid])

    return train_samples, val_samples


def _load_test_samples(test_dir=DATA_TEST_PATH):
    """
    Loads all test samples from test directory.
    
    Returns:
        list: List of (image_path, label) tuples.
    """
    test_samples = []
    classes = [("NC", 0), ("AD", 1)]

    for class_name, label in classes:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.exists(class_dir):
            continue

        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpeg", ".jpg")):
                img_path = os.path.join(class_dir, fname)
                test_samples.append((img_path, label))

    return test_samples
