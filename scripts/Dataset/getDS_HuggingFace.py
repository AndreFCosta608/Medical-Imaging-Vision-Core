import os
from datasets import load_dataset

dataset_name = 'Onegafer/vehicle_segmentation'

split = "train"  # ou "test", "validation", etc.
print(f"🔽 Baixando o dataset: {dataset_name}...")
dataset = load_dataset(dataset_name, split=split)

print(f"✅ Dataset carregado com {len(dataset)} amostras.")
print("Exemplo:", dataset[0])

output_dir = '/home/pi/Deposito/Projetos/Meus/CertificacaoHuggingFace/fontes/TempDataSet/'
os.makedirs(output_dir, exist_ok=True)
dataset.save_to_disk(os.path.join(output_dir, dataset_name.replace("/", "_")))

print(f"💾 Dataset salvo em: {output_dir}/{dataset_name.replace('/', '_')}")
