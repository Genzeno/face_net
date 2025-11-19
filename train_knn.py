from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import joblib

train = np.load("embeddings_train.npz", allow_pickle=True)
X, y = train["X"], train["y"]

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

joblib.dump(knn, "facenet_knn.joblib")
print("Saved as facenet_knn.joblib")
