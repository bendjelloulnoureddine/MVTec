import os
import cv2
from glob import glob
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from config.config import Config as C

class MVTecDataset(Dataset):
    def __init__(self, root_dir):
        self.files = sorted(glob(os.path.join(root_dir, "*.png")))
        self.transform = transforms.Compose([
            transforms.Resize((C.IMAGE_SIZE, C.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = cv2.imread(self.files[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        return self.transform(img)

def get_train_loader():
    return DataLoader(
        MVTecDataset("dataset/screw/train/good"), 
        batch_size=C.BATCH_SIZE, 
        shuffle=False,
        num_workers=2,  # Reduced for memory efficiency
        pin_memory=False,  # Disable pin_memory to save GPU memory
        persistent_workers=False  # Disable to save memory
    )

def get_test_images():
    test_files = sorted(glob("dataset/screw/test/**/*.png", recursive=True))
    return test_files

