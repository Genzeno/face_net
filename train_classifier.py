from sklearn.svm import SVC
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# load embeddings
train = np.load("embeddings_train.npz", allow_pickle=True)
X, y = train["X"], train["y"]

# encode label
label_encoder = LabelEncoder()
y_enc = label_encoder.fit_transform(y)

print("Training SVM...")
clf = SVC(kernel='linear', probability=True)
clf.fit(X, y_enc)

# save classifier + encoder
joblib.dump((clf, label_encoder), "facenet_svm.joblib")

print("Saved (clf, label_encoder) → facenet_svm.joblib")
