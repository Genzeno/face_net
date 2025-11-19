import os
import joblib
import numpy as np
from utils_facenet import extract_embedding

model = joblib.load("facenet_svm.joblib")

VAL = "data/val/"

for class_name in os.listdir(VAL):
    folder = os.path.join(VAL, class_name)

    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        emb = extract_embedding(img_path)

        if emb is None:
            continue

        pred = model.predict([emb])[0]
        print(f"{img_path} → predicted: {pred}, actual: {class_name}")
