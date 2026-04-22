#encoding=utf8
import os
from PIL import Image
import scripts.config as config

def resize_all(target, extensao):
    paths = [os.path.join(target, nome) for nome in os.listdir(target)]
    files = [arq for arq in paths if os.path.isfile(arq)]
    files = [arq for arq in files if arq.lower().endswith(extensao)]
    for img in files:
        imagePath = str(img)
        print('   ' + imagePath)
        try:
            image = Image.open(imagePath)
            resized = image.resize((config.width, config.height), Image.LANCZOS)
            resized.save(imagePath)
        except Exception as e:
            print(f" {imagePath}: {e}")

print('Starting processing...')

resize_all(config.images, '.jpg')
resize_all(config.masks, '.png')

print('\n\nCompleted...')
