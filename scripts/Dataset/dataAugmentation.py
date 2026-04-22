import os
import cv2
import numpy as np
from PIL import Image
import albumentations as A
import scripts.config as config

input_img_dir = config.images
out_img_dir = config.images #+ "/images_aug"

input_mask_dir = config.masks
out_mask_dir = config.masks #+ "/masks_aug"

print('Starting...')

os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_mask_dir, exist_ok=True)

transformations = [
    ("flip", A.HorizontalFlip(p=1.0)),
    ("rot15", A.Rotate(limit=15, p=1.0)),
    ("contrast", A.RandomBrightnessContrast(p=1.0)),
]

for fname in os.listdir(input_img_dir):
    if not fname.endswith(".jpg"): continue
    base = fname.replace(".jpg", "")
    
    img_path = os.path.join(input_img_dir, fname)
    mask_path = os.path.join(input_mask_dir, base + "_mask.png")

    img = np.array(Image.open(img_path).convert("L"))
    mask = np.array(Image.open(mask_path).convert("L"))
    mask = (mask > 127).astype('uint8')

    for name, tf in transformations:
        aug = A.Compose([tf])
        augmented = aug(image=img, mask=mask)
        img_aug = augmented['image']
        mask_aug = augmented['mask']

        img_out = os.path.join(out_img_dir, f"{base}_aug_{name}.jpg")
        mask_out = os.path.join(out_mask_dir, f"{base}_aug_{name}_mask.png")

        cv2.imwrite(img_out, img_aug)
        cv2.imwrite(mask_out, mask_aug * 255)
        
print('Completed...')        
