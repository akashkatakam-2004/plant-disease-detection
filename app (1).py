"""
Plant Disease Detection - Hugging Face Spaces
VGG16 CNN Feature Extractor + XGBoost Classifier
"""

import os
import numpy as np
import cv2
import joblib
import gradio as gr
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers

IMG_SIZE   = (224, 224)
MODELS_DIR = "."

def create_feature_extractor(input_shape=(224, 224, 3)):
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation="relu", name="feature_layer_1"),
        layers.Dropout(0.3),
        layers.Dense(256, activation="relu", name="feature_layer_2"),
        layers.Dropout(0.2),
        layers.Dense(128, activation="relu", name="features"),
    ])
    return model

print("Loading models...")
xgb_model   = joblib.load(os.path.join(MODELS_DIR, "xgb_classifier.pkl"))
metadata    = joblib.load(os.path.join(MODELS_DIR, "metadata.pkl"))
class_names = metadata["class_names"]

cnn_path = os.path.join(MODELS_DIR, "cnn_extractor.h5")
if os.path.exists(cnn_path):
    feature_extractor = keras.models.load_model(cnn_path)
    print("Loaded saved CNN extractor")
else:
    feature_extractor = create_feature_extractor()
    print("WARNING: cnn_extractor.h5 not found - using ImageNet weights only")

print(f"Ready! Classes: {class_names}")

DISEASE_INFO = {
    "Apple___Apple_scab":                        "Apple Scab is a fungal disease causing dark, scaly lesions on leaves and fruit.",
    "Apple___Black_rot":                         "Black Rot causes circular lesions on leaves and rotting of fruit.",
    "Apple___healthy":                           "The apple plant looks healthy with no signs of disease.",
    "Cherry_(including_sour)___Powdery_mildew":  "Powdery Mildew appears as white powdery spots on leaves.",
    "Cherry_(including_sour)___healthy":         "The cherry plant looks healthy with no signs of disease.",
    "Grape___Black_rot":                         "Grape Black Rot causes brown lesions on leaves and shriveled fruit.",
    "Grape___healthy":                           "The grape plant looks healthy with no signs of disease.",
    "Tomato___Late_blight":                      "Late Blight is a serious disease causing dark lesions on leaves and stems.",
    "Tomato___Leaf_Mold":                        "Leaf Mold causes yellow spots on upper leaf surface and mold below.",
    "Tomato___healthy":                          "The tomato plant looks healthy with no signs of disease.",
}

def predict(image):
    if image is None:
        return "Please upload an image.", {}, ""
    img = np.array(image.convert("RGB"))
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    img_batch = np.expand_dims(img, axis=0)
    features      = feature_extractor.predict(img_batch, verbose=0)
    prediction    = xgb_model.predict(features)[0]
    probabilities = xgb_model.predict_proba(features)[0]
    predicted_idx   = int(prediction)
    predicted_class = class_names[predicted_idx]
    confidence      = float(probabilities[predicted_idx])
    is_healthy      = "healthy" in predicted_class.lower()
    parts   = predicted_class.split("___")
    plant   = parts[0].replace("_", " ").split("(")[0].strip()
    disease = parts[1].replace("_", " ").title() if len(parts) > 1 else predicted_class
    status  = "✅ HEALTHY" if is_healthy else "⚠️ DISEASE DETECTED"
    top3_idx  = np.argsort(probabilities)[-3:][::-1]
    top3_dict = {
        class_names[i].replace("___", " — ").replace("_", " "): float(probabilities[i])
        for i in top3_idx
    }
    info = DISEASE_INFO.get(predicted_class, "")
    summary = f"""**{status}**

🌱 **Plant:** {plant}
🔬 **Condition:** {disease}
📊 **Confidence:** {confidence*100:.1f}%

{info}"""
    return summary, top3_dict, predicted_class

with gr.Blocks(
    title="Plant Disease Detection",
    theme=gr.themes.Soft(primary_hue="green"),
) as demo:
    gr.Markdown("# 🌿 Plant Disease Detection")
    gr.Markdown("Upload a plant leaf image to detect diseases using **VGG16 CNN + XGBoost**.")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Leaf Image")
            predict_btn = gr.Button("Analyse Plant", variant="primary", size="lg")
        with gr.Column(scale=1):
            result_text      = gr.Markdown(label="Result")
            confidence_chart = gr.Label(label="Top Predictions", num_top_classes=3)
            predicted_label  = gr.Textbox(label="Raw Class", visible=False)

    gr.Markdown("""---
### Supported Plants
| Plant | Conditions |
|-------|-----------|
| Apple | Apple Scab, Black Rot, Healthy |
| Cherry | Powdery Mildew, Healthy |
| Grape | Black Rot, Healthy |
| Tomato | Late Blight, Leaf Mold, Healthy |
""")

    predict_btn.click(fn=predict, inputs=image_input, outputs=[result_text, confidence_chart, predicted_label])
    image_input.change(fn=predict, inputs=image_input, outputs=[result_text, confidence_chart, predicted_label])

if __name__ == "__main__":
    demo.launch()
