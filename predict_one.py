from utils_facenet import extract_embedding, load_classifier

img_path = "data/val/stevano/stevano1.jpg"

clf, label_encoder = load_classifier()

emb = extract_embedding(img_path)
pred = clf.predict([emb])[0]

name = label_encoder.inverse_transform([pred])[0]

print("Predicted:", name)
