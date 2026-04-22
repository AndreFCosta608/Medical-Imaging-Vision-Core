import os
import torch
import numpy as np
from PIL import Image
import scripts.config as config
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class SegmentationDataset(Dataset):

    def __init__(self, transform=None):
        self.image_dir = config.images
        self.mask_dir = config.masks
        self.transform = transform
        paths = [os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir) if f.lower().endswith('.jpg')]
        self.image_files = [os.path.basename(f) for f in paths]
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '_mask.png'))
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found for: {img_name}")

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)

        mask = np.array(mask)
        mask = (mask > 127).astype(np.uint8)
        mask = torch.from_numpy(mask).long()
        
        unique_vals = np.unique(mask)
        if not set(unique_vals).issubset({0, 1}):
            raise ValueError(f"Mask contains invalid values: {unique_vals}")

        return image, mask
