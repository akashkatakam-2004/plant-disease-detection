# 🌿 Plant Disease Detection

> Upload a plant leaf image → instantly detect diseases using **VGG16 CNN + XGBoost**.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange?logo=tensorflow)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-green)
![Gradio](https://img.shields.io/badge/Gradio-UI-purple)
![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace%20Space-yellow)

---

## 🚀 Live Demo
👉 [Try it on Hugging Face Spaces](https://huggingface.co/spaces/Akashkatakam/plant-disease-detection)

---

## ✨ Features

- 🖼️ **Image Upload** — Upload any plant leaf image
- 🧠 **VGG16 Feature Extraction** — Deep CNN extracts 128-dim feature vectors
- 🌲 **XGBoost Classifier** — Fast, accurate disease classification
- 📊 **Top 3 Predictions** — See confidence scores for top predictions
- ✅ **Healthy vs Disease** — Clear status with detailed disease info
- 🌱 **10 Conditions** — Covers 4 plants and 10 disease/healthy states

---

## 🏗️ Architecture

```
Leaf Image (224x224)
       │
       ▼
VGG16 (ImageNet weights, frozen)
       │
GlobalAveragePooling2D
       │
Dense(512) → Dense(256) → Dense(128)
       │
  Feature Vector (128-dim)
       │
       ▼
XGBoost Classifier
       │
       ▼
Disease Prediction + Confidence
```

---

## 🌱 Supported Plants & Conditions

| Plant | Conditions |
|-------|-----------|
| 🍎 Apple | Apple Scab, Black Rot, Healthy |
| 🍒 Cherry | Powdery Mildew, Healthy |
| 🍇 Grape | Black Rot, Healthy |
| 🍅 Tomato | Late Blight, Leaf Mold, Healthy |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Gradio |
| Feature Extractor | VGG16 (TensorFlow/Keras) |
| Classifier | XGBoost |
| Image Processing | OpenCV + PIL |
| Model Storage | Hugging Face Spaces (LFS) |

---

## ⚙️ Local Setup

### 1. Clone the repo
```bash
git clone https://github.com/akashkatakam-2004/plant-disease-detection.git
cd plant-disease-detection
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the CNN model
> ⚠️ `cnn_extractor.h5` (60MB) is NOT included in this repo due to GitHub's 25MB limit.

Download it from Hugging Face:
👉 [Download cnn_extractor.h5](https://huggingface.co/spaces/Akashkatakam/plant-disease-detection/resolve/main/cnn_extractor.h5)

Place it in the root folder:
```
plant-disease-detection/
├── cnn_extractor.h5   ← place here
├── xgb_classifier.pkl
├── metadata.pkl
├── app.py
...
```

### 5. Run the app
```bash
python app.py
```

Open your browser at `http://localhost:7860`

---

## 📂 Project Structure

```
plant-disease-detection/
├── app.py               # Gradio app + prediction logic
├── xgb_classifier.pkl   # Trained XGBoost model
├── metadata.pkl         # Class names and metadata
├── requirements.txt     # Python dependencies
├── .gitignore           # Excludes large model files
└── README.md            # You are here

# Not in repo (too large for GitHub):
# cnn_extractor.h5      # Download from Hugging Face (link above)
```

---

## 👨‍💻 Built by

**Akash Katakam** — [Hugging Face](https://huggingface.co/Akashkatakam)

---

## 📄 License

MIT License — feel free to use, modify, and share!
