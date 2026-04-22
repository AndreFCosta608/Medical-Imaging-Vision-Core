#encoding=utf8
import os
import pandas as pd
import scripts.config as config
from sklearn.model_selection import train_test_split

nameTrain = config.source + 'train.txt'
nameVal = config.source + 'val.txt'
percentual = 0.7

def nomeOrigem(originalName):
    nome = os.path.join(config.images, originalName)
    return nome

def nomeMascara(originalName):
    nome = os.path.join(config.masks, originalName)
    nome = nome.replace('.jpg', '.png')
    return nome

print('Starting processing...')
paths = [os.path.join(config.images, nome) for nome in os.listdir(config.images)]
files = [arq for arq in paths if os.path.isfile(arq)]
files = [arq for arq in files if arq.lower().endswith('.jpg')]

df = pd.DataFrame({'nome_original': files})

df['newNameA'] = [nomeOrigem(nome) for i, nome in enumerate(files)]
df['newNameB'] = [nomeMascara(nome) for i, nome in enumerate(files)]

grupo1, grupo2 = train_test_split(df, train_size=percentual, random_state=42)

grupo1[['newNameA', 'newNameB']].to_csv(nameTrain, sep='\t', index=False, header=False)
grupo2[['newNameA', 'newNameB']].to_csv(nameVal, sep='\t', index=False, header=False)

print("Groups saved in '" + nameTrain + "' and '" + nameVal + "'")
