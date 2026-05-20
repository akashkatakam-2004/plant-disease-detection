# 🌿 Plant Disease Detection

> Upload a plant leaf image → instantly detect diseases using **VGG16 CNN + XGBoost**.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange?logo=tensorflow)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-green)
![Gradio](https://img.shields.io/badge/Gradio-UI-purple)
![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace%20Space-yellow)

## 🚀 Live Demo
👉 [Try it on Hugging Face Spaces](https://huggingface.co/spaces/Akashkatakam/plant-disease-detection)

## ✨ Features

- 🖼️ **Image Upload** — Upload any plant leaf image
- 🧠 **VGG16 Feature Extraction** — Deep CNN extracts 128-dim feature vectors
- 🌲 **XGBoost Classifier** — Fast, accurate disease classification
- 📊 **Top Prediction** — See disease name + confidence score
- ✅ **Healthy vs Disease** — Clear status with detailed disease info
- 🌱 **38 Classes** — Covers 14 plants (Apple, Tomato, Potato, Grape, Corn, and more)

## 🏗️ Architecture
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

text

## 🌱 Supported Plants & Diseases

| Plant | Diseases |
|-------|----------|
| 🍎 Apple | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| 🫐 Blueberry | Healthy |
| 🍒 Cherry | Powdery Mildew, Healthy |
| 🌽 Corn | Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| 🍇 Grape | Black Rot, Esca, Leaf Blight, Healthy |
| 🍊 Orange | Citrus Greening (Haunglongbing) |
| 🍑 Peach | Bacterial Spot, Healthy |
| 🫑 Pepper | Bacterial Spot, Healthy |
| 🥔 Potato | Early Blight, Late Blight, Healthy |
| 🍓 Strawberry | Leaf Scorch, Healthy |
| 🍅 Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |
| + Raspberry, Soybean, Squash | Healthy + Powdery Mildew (Squash) |

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Gradio |
| Feature Extractor | VGG16 (TensorFlow/Keras) |
| Classifier | XGBoost |
| Image Processing | OpenCV + PIL |
| Deployment | Hugging Face Spaces |

## ⚙️ Local Setup

```bash
git clone https://github.com/akashkatakam-2004/plant-disease-detection.git
cd plant-disease-detection
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
Open http://localhost:7860

⚠️ The CNN feature extractor (cnn_extractor.h5) creates automatically on first run.

📂 Project Structure
text
plant-disease-detection/
├── app.py               # Gradio app
├── xgb_classifier.pkl   # Trained XGBoost model
├── metadata.pkl         # Class names
├── requirements.txt     # Dependencies
└── README.md
📊 Model Performance
Metric	Value
Validation Accuracy	~96%
Test Accuracy	~95%
Number of Classes	38
👨‍💻 Built by
Akash Katakam — AI/ML Engineer

GitHub | LinkedIn | Hugging Face
