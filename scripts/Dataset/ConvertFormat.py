#encoding=utf8
import os
from PIL import Image
import scripts.config as config

target = config.masks
extension_destination = '.png'

print('Starting processing...')
files = os.listdir(target)
for fileOne in files:
    nameFile, extension = os.path.splitext(fileOne)
    print('Testing: ', fileOne)
    if((extension != extension_destination) and 
       (extension != '.json')):
        if(extension == ''):
            extension = '....'

        print('\nRenaming: ' + str(fileOne))
        origem = os.path.join(target, fileOne)
        
        destino = fileOne.replace(extension, extension_destination)
        if(extension == '....'):
            destino = destino + extension_destination
        destino = os.path.join(target, destino)
    
        im = Image.open(origem).convert('RGB')
        im.save(destino, 'PNG')
        os.remove(origem)
        print('-----------------------------------------')
    
print('\n\nCompleted...')
