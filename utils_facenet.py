import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import cv2
import os


# Load FaceNet model + MTCNN detector
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=160, margin=20, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def extract_embedding(img_path):
    img = Image.open(img_path)

    face = mtcnn(img)
    if face is None:
        print("⚠️ No face detected in:", img_path)
        return None

    embedding = resnet(face.unsqueeze(0).to(device)).detach().cpu().numpy()
    return embedding[0]


def load_images_from_folder(folder):
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                paths.append(os.path.join(root, f))
    return paths

import joblib
import numpy as np
from PIL import Image

def load_classifier(model_path="facenet_svm.joblib"):
    return joblib.load(model_path)  # returns (clf, label_encoder)
