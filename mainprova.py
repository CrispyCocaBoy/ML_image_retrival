import os
import torch
import json
import requests
import numpy as np
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.metrics.pairwise import cosine_similarity

# === Submit function ===
def submit(results, groupname, url="http://tatooine.disi.unitn.it:3001/retrieval/"):
    res = {"groupname": groupname, "images": results}
    res = json.dumps(res)
    response = requests.post(url, res)
    try:
        result = json.loads(response.text)
        print(f"Accuracy is {result['accuracy']}")
    except json.JSONDecodeError:
        print(f"ERROR: {response.text}")

# === Paths (update here) ===
train_data_root = "Competition/train"  # se in futuro vorrai fare training
query_data_root = "Competition/test/query"
gallery_data_root = "Competition/test/gallery"
group_name = "YOUR_GROUP_NAME"  # <-- Inserisci qui il nome del tuo gruppo

# === Device and models ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=0, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# === Embedding extraction ===
def extract_embedding(img_path):
    try:
        img = Image.open(img_path).convert('RGB')
        face = mtcnn(img)
        if face is None:
            return None
        with torch.no_grad():
            embedding = model(face.unsqueeze(0).to(device))
        return embedding.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error with image {img_path}: {e}")
        return None

# === Step 1: Compute gallery embeddings ===
print("Extracting gallery embeddings...")
gallery_embeddings = {}
for filename in os.listdir(gallery_data_root):
    path = os.path.join(gallery_data_root, filename)
    emb = extract_embedding(path)
    if emb is not None:
        gallery_embeddings[filename] = emb

# === Step 2: For each query, retrieve top 10 similar ===
print("Processing queries...")
results = {}
for filename in os.listdir(query_data_root):
    query_path = os.path.join(query_data_root, filename)
    query_emb = extract_embedding(query_path)
    if query_emb is None:
        print(f"Warning: No face found in query {filename}")
        continue

    similarities = []
    for gal_name, gal_emb in gallery_embeddings.items():
        sim = cosine_similarity([query_emb], [gal_emb])[0][0]
        similarities.append((gal_name, sim))

    # Sort gallery images by similarity (descending)
    top10 = sorted(similarities, key=lambda x: -x[1])[:10]
    results[filename] = [name for name, _ in top10]

# === Step 3: Submit the results ===
submit(results, group_name="Simple Guys")
