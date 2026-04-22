import os
import cv2
import numpy as np
from PIL import Image
import scripts.config as config
import matplotlib.pyplot as plt

print('Starting processing...')
paths = [os.path.join(config.masks, nome) for nome in os.listdir(config.masks)]
files = [arq for arq in paths if os.path.isfile(arq)]
masks = [arq for arq in files if arq.lower().endswith('.png')]
for maskName in masks:
    
    imgName = str(maskName)
    imgName = imgName.replace(config.masks, '')
    imgName = imgName.replace('.png', '.jpg')
    if('_' in imgName):
        vetImgName = imgName.split('_')
        imgName = vetImgName[0] + '.jpg'
    
    print('imgName = ', str(imgName).replace(config.images, ''))
    print('maskName = ', str(maskName).replace(config.masks, ''))

    imgPath = os.path.join(config.images, imgName)
    img = Image.open(imgPath).convert("L").resize((config.height, config.width))
    img_np = np.array(img)

    maskPath = os.path.join(config.masks, maskName)
    mask = Image.open(maskPath).resize((config.height, config.width)).convert("L")
    mask_np = np.array(mask) // 255

    overlay = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    overlay[mask_np == 1] = [255, 0, 0]

    output = cv2.addWeighted(cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR), 0.7, overlay, 0.3, 0)

    plt.figure(figsize=(6, 6))
    plt.imshow(output)
    plt.title(f"Overlay")
    plt.axis("off")
    plt.show()

    input("Press ENTER to view the next...")

print('\n\nCompleted...')
