import os 

device = ''
width = 512
height = 512

num_epochs = 100
checkpoint_interval = 2
early_stop_patience = 10

USE_TTA = True
USE_REFINEMENT = True
USE_FOCAL_LOSS = False

source = str(os.path.dirname(os.path.realpath(__file__))).replace('scripts', '')
images = source + 'DataSet/images/'
masks = source + 'DataSet/masks/'
annotations = source + 'DataSet/annotations/'
extraTests = source + 'DataSet/ExtraTests/'
tempImages = source + 'DataSet/tempImages/'
checkpoints = source + 'checkpoints/'

modelName = checkpoints + 'modelo_completo.pth'# 'modelo_completo.pth'
report_file = source + 'report_file.txt'

