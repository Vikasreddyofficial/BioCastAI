import os
import faiss
import pickle
import numpy as np
import streamlit as st
from mtcnn import MTCNN
from keras_facenet import FaceNet
from PIL import Image
import cv2

# Load FAISS index & actor names
index = faiss.read_index("data/faiss_index.bin")
with open("data/actor_names.pkl", "rb") as f:
    actor_names = pickle.load(f)

# Initialize FaceNet & MTCNN
embedder = FaceNet()
detector = MTCNN()

def detect_align_face(image):
    """Detect & align face from uploaded image."""
    img_rgb = np.array(image.convert("RGB"))  # Convert PIL image to NumPy
    faces = detector.detect_faces(img_rgb)
    
    if len(faces) == 0:
        return None

    x, y, width, height = faces[0]['box']
    face_crop = img_rgb[y:y+height, x:x+width]

    # Resize to 160x160 for FaceNet
    aligned_face = cv2.resize(face_crop, (160, 160))
    return aligned_face

def get_face_embedding(image):
    """Extract 512D face embedding."""
    face = detect_align_face(image)
    if face is None:
        return None

    face = np.expand_dims(face, axis=0)
    return embedder.embeddings(face)[0]

def find_best_match(image):
    """Find the best matching actor."""
    query_embedding = get_face_embedding(image)
    if query_embedding is None:
        return None  # No face detected

    # Convert query embedding to required format
    query_embedding = np.array([query_embedding], dtype="float32")

    # Search for the closest match in FAISS
    distances, indices = index.search(query_embedding, k=3)  # Get top 3 matches
    
    # Retrieve matching actor names
    recommendations = [(actor_names[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
    return recommendations

# Streamlit UI
st.title("🎭 BioCastAI: Biopic Casting Recommendation")
st.write("Upload an image, and we'll recommend the best-matching actors.")

uploaded_image = st.file_uploader("📤 Upload an image of a real-life character", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Finding best actor matches..."):
        matches = find_best_match(image)
    
    if matches:
        st.subheader("Recommended Actors:")
        for i, (actor, score) in enumerate(matches):
            actor_folder = f"actors_dataset/Indian_actors_faces/{actor}"
            
            # Get the first available image in the actor's folder
            if os.path.exists(actor_folder):
                actor_images = [img for img in os.listdir(actor_folder) if img.endswith(('.jpg', '.png', '.jpeg'))]
                
                if actor_images:
                    actor_img_path = os.path.join(actor_folder, actor_images[0])
                    actor_img = Image.open(actor_img_path)
                    st.image(actor_img, caption=f"{actor} (Score: {score:.4f})", width=200)
                else:
                    st.write(f"{i+1}. {actor} (Score: {score:.4f}) - [No Image Available]")
            else:
                st.write(f"{i+1}. {actor} (Score: {score:.4f}) - [Folder Not Found]")
    else:
        st.error("No face detected! Try another image.")
