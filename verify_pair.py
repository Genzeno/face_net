import numpy as np
from utils_facenet import extract_embedding
from numpy.linalg import norm


img1 = "a.jpg"
img2 = "b.jpg"

emb1 = extract_embedding(img1)
emb2 = extract_embedding(img2)

dist = norm(emb1 - emb2)

print("Distance:", dist)

if dist < 1.0:
    print("Same person")
else:
    print("Different persons")
