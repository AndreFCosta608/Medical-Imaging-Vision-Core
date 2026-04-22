import os
import scripts.config as config
from simple_image_download import simple_image_download as simp

termo_busca = "garrote"
pasta_destino = config.tempImages
quantidade = 200


if not os.path.exists(pasta_destino):
    os.makedirs(pasta_destino)
    
response = simp.simple_image_download
response().download(termo_busca, quantidade)


print(f"Download complete! Images saved to: {pasta_destino}")
