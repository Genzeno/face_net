import os
import numpy as np
from utils_facenet import extract_embedding, load_images_from_folder


DATASET = "data/train/"
OUT_FILE = "embeddings_train.npz"

X, y = [], []

for class_name in os.listdir(DATASET):
    folder = os.path.join(DATASET, class_name)
    if not os.path.isdir(folder):
        continue

    print("Processing:", class_name)
    images = load_images_from_folder(folder)

    for img in images:
        emb = extract_embedding(img)
        if emb is not None:
            X.append(emb)
            y.append(class_name)

np.savez(OUT_FILE, X=X, y=y)
print("Done! File saved:", OUT_FILE)
