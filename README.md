# 🎭 BioCastAI: Biopic Casting Recommendation

## 📌 Overview
BioCastAI is an AI-powered face recognition system that recommends actors for real-life characters in biopics. Given an image, it finds the best-matching actors from a predefined dataset using deep learning and FAISS-based similarity search.

## 🚀 Features
- 🎭 **Actor Matching:** Finds the closest matching actors for a given face.
- 📸 **Face Detection & Embedding:** Uses MTCNN for face detection and FaceNet for generating 512D embeddings.
- ⚡ **FAISS Indexing:** Efficient similarity search with FAISS.
- 🖼️ **Image-Based Recommendations:** Displays matched actors' images along with similarity scores.
- 📊 **Streamlit UI:** Interactive web interface for easy usage.

## 🛠️ Tech Stack
- **Python** (Core development)
- **TensorFlow & Keras** (FaceNet embeddings)
- **MTCNN** (Face detection)
- **FAISS** (Efficient similarity search)
- **OpenCV & PIL** (Image processing)
- **Streamlit** (Web-based UI)

## 📂 Project Structure
```
BioCastAI/
│── actors_dataset/              # Actor image dataset (Not included in GitHub)
│── data/
│   ├── faiss_index.bin          # Precomputed FAISS index
│   ├── actor_names.pkl          # Pickled actor name mappings
│── venv/                        # Virtual environment (Ignored in Git)
│── app.py                       # Main Streamlit application
│── requirements.txt             # List of dependencies
│── README.md                    # Documentation
│── .gitignore                    # Ignoring unnecessary files
```

## 🔧 Installation & Setup
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/Vikasreddyofficial/BioCastAI.git
cd BioCastAI
```

### 2️⃣ Create & Activate Virtual Environment
```sh
python -m venv venv  # Create a virtual environment
source venv/bin/activate  # Activate (Linux/macOS)
venv\Scripts\activate  # Activate (Windows)
```

### 3️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 4️⃣ Run the Application
```sh
streamlit run app.py
```

## 🖼️ Usage
1️⃣ Upload an image of a real-life character.
2️⃣ BioCastAI detects the face and extracts embeddings.
3️⃣ It searches for the closest matching actors using FAISS.
4️⃣ Recommended actors are displayed with similarity scores and images.

## 📌 Notes
- **Dataset Storage:** Actor images are stored in `actors_dataset/` (not included in GitHub due to size constraints).
- **Large File Handling:** We have ignored large model files & dependencies in `.gitignore`.

## 📜 License
This project is licensed under the **MIT License**.

---
💡 **Developed by Vikas Reddy** 🚀

