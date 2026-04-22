#encoding=utf8
import os
from PIL import Image
import scripts.config as config

def convert_all(target, extensao):
    paths = [os.path.join(target, nome) for nome in os.listdir(target)]
    files = [arq for arq in paths if os.path.isfile(arq)]
    arquivos = [arq for arq in files if arq.lower().endswith(extensao)]
    for img in arquivos:
        pathImage = str(img)
        print('   ' + pathImage)
    
        image = Image.open(pathImage)
        image = image.convert("L")
        image.save(pathImage)

print('Starting processing...')

convert_all(config.images, '.jpg')
convert_all(config.masks, '.png')

print('\n\nCompleted...')
