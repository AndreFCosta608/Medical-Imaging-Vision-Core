import numpy as np
from PIL import Image

path = '/home/pi/Deposito/Projetos/Meus/CertificacaoHuggingFace/fontes/'
path = path + 'DataSet/masks/0cdf5b5d0ce1_01_mask.png'

mask = Image.open(path).convert("L")
print(np.unique(np.array(mask)))