import os
import cv2
import json
import numpy as np
from PIL import Image
import scripts.config as config

print('Starting processing...')
os.makedirs(config.masks, exist_ok=True)
for nameFile in os.listdir(config.images):
    if nameFile.endswith('.json'):
        jsonPath = os.path.join(config.images, nameFile)

        dados = ''
        with open(jsonPath, 'r', encoding='utf-8') as f:
            dados = json.load(f)

        baseName = os.path.splitext(nameFile)[0]
        imagePath = os.path.join(config.images, baseName + '.jpg')

        image = cv2.imread(imagePath)
        if image is None:
            print(f"Image {imagePath} not found or invalid.")
            continue

        height, width = image.shape[:2]
        imgShape = (height, width)
         
        contador = 0
        for shape in dados.get('shapes', []):
            if shape.get('label') == 'gado' and shape.get('shape_type') == 'polygon':
                mask = np.zeros(imgShape, dtype=np.uint8)
                pts = np.array(shape['points'], dtype=np.int32)
                cv2.fillPoly(mask, [pts], color=1)
        
                imgMask = (mask * 255).astype(np.uint8)
                imgMask = Image.fromarray(imgMask)
        
                maskName = f"{baseName}_{contador}.png" if contador > 1 else f"{baseName}.png"
                endPath = os.path.join(config.masks, maskName)
        
                imgMask.save(endPath)
                print(f"Saved: {endPath}")
        
                contador += 1

print('\n\nCompleted...')
