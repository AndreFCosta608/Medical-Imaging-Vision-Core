import os
import imagehash
from PIL import Image
import scripts.config as config

hashes = {}
duplicates = []

paths = [os.path.join(config.images, nome) for nome in os.listdir(config.images)]
files = [arq for arq in paths if os.path.isfile(arq)]
fileNames = [arq for arq in files if arq.lower().endswith('.jpg')]
for filename in fileNames:
    filepath = filename
    try:
        with Image.open(filepath) as img:
            img_hash = imagehash.phash(img)  # Ou: dhash(img), average_hash(img)
            if img_hash in hashes:
                print(f"Duplicate detected: {filename} ≈ {hashes[img_hash]}")
                duplicates.append(filepath)
            else:
                hashes[img_hash] = filename
    except Exception as e:
        print(f"Error processing {filename}: {e}")

for dup in duplicates:
    try:
        os.remove(dup)
        print(f"Removed: {dup}")
    except Exception as e:
        print(f"Error removing {dup}: {e}")

print(f"\nTotal duplicates removed: {len(duplicates)}")
