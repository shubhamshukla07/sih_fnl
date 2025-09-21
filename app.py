import streamlit as st
import torch
import clip
from PIL import Image
import numpy as np
import hashlib
import os

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
HASH_FILE = "processed_hashes.txt"

CLASS_LABELS = [
    "Plastic waste - complaint sent to Waste Dept.",
    "Paper waste - complaint sent to Waste Dept.",
    "Metal waste - complaint sent to Waste Dept.",
    "Organic waste - complaint sent to Waste Dept.",
    "Overflowing garbage - complaint sent to Sanitation Dept.",
    "Public washroom issue - complaint sent to Health Dept.",
    "Pothole - complaint sent to Road Dept.",
    "Damaged road - complaint sent to Public Works Dept.",
    "Broken streetlight - complaint sent to Electrical Dept.",
    "Exposed wires - complaint sent to Electrical Safety Dept.",
    "Leaking pipe - complaint sent to Water Dept.",
    "Dirty water - complaint sent to Water Quality Dept.",
    "Vandalism - complaint sent to Security Dept.",
    "Damaged park - complaint sent to Parks Dept."
]

# Hashing utilities
def get_image_hash(image_bytes):
    return hashlib.md5(image_bytes).hexdigest()

def load_hashes():
    if not os.path.exists(HASH_FILE):
        return set()
    with open(HASH_FILE, "r") as f:
        return set(line.strip() for line in f)

def save_hash(image_hash):
    with open(HASH_FILE, "a") as f:
        f.write(image_hash + "\n")

# Streamlit UI
st.title("🧠ViT-B/32 Smart Waste Classifier")
st.write("Upload an image to classify the type of issue and route the complaint.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_bytes = uploaded_file.read()
    image_hash = get_image_hash(image_bytes)
    processed_hashes = load_hashes()

    if image_hash in processed_hashes:
        st.warning("⚠️ This image has already been processed. Duplicate entry not allowed.")
    else:
        save_hash(image_hash)
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        image_input = preprocess(image).unsqueeze(0).to(device)
        text_inputs = clip.tokenize(CLASS_LABELS).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            text_features = model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = (image_features @ text_features.T).squeeze(0)
            probs = similarity.softmax(dim=0).cpu().numpy()

        best_idx = np.argmax(probs)
        predicted_label = CLASS_LABELS[best_idx]
        confidence = probs[best_idx]

        st.success(f"✅ Predicted Label: **{predicted_label}**")
        st.metric(label="Confidence", value=f"{confidence:.4f}")