#encoding=utf8
import os
import scripts.config as config

extension = '.jpg'

print('Starting processing...')
contador = 0
paths = [os.path.join(config.images, nome) for nome in os.listdir(config.images)]
files = [arq for arq in paths if os.path.isfile(arq)]
jpgs = [arq for arq in files if arq.lower().endswith(extension)]
for img in jpgs:
    imagePath = str(img)
    
    oldName = imagePath.replace(config.images, '')
    print('\n   ' + oldName)
    
    newName = config.images + str(contador) + extension
    os.rename(imagePath, newName)
    contador = contador + 1

print('\n\nCompleted...')
