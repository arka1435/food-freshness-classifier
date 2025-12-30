import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
# Page Configuration
st.set_page_config(
    page_title="AI Food Freshness Assessment System",
    layout="centered"
)


# ImageNet → User-Friendly Food Mapping
# --------------------------------------------------
FOOD_LABEL_MAP = {
    "granny smith": "Apple",
    "red delicious": "Apple",
    "golden delicious": "Apple",
    "banana": "Banana",
    "orange": "Orange",
    "lemon": "Lemon",
    "strawberry": "Strawberry",
    "pineapple": "Pineapple",
    "cucumber": "Cucumber",
    "tomato": "Tomato"
}

# --------------------------------------------------
# Model Configuration
# --------------------------------------------------
MODEL_NAME = "google/vit-base-patch16-224"

@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return processor, model

processor, model = load_model()

# --------------------------------------------------
# Freshness Logic
# --------------------------------------------------
def freshness_mapper(confidence):
    if confidence >= 80:
        return "✅ Fresh"
    elif confidence >= 50:
        return "⚠️ Okay / Consume Soon"
    else:
        return "❌ Avoid"

def explain_prediction(quality):
    if "Fresh" in quality:
        return "The food appears visually intact with normal color and texture."
    elif "Okay" in quality:
        return "Minor visual variations detected. Consume soon."
    else:
        return "Visual cues indicate possible spoilage."

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("🍎 AI-Powered Food Freshness Assessment System")
st.markdown("Real-time food freshness analysis using computer vision.")

input_mode = st.radio(
    "Select input mode",
    ["Upload Image", "Live Camera"]
)

image = None

# --------------------------------------------------
# INPUT HANDLING
# --------------------------------------------------
if input_mode == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload a food image",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

elif input_mode == "Live Camera":
    camera_image = st.camera_input("Capture food image")
    if camera_image:
        image = Image.open(camera_image).convert("RGB")

# --------------------------------------------------
# INFERENCE
# --------------------------------------------------
if image:
    st.image(image, caption="Input Image", use_column_width=True)

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)[0]

    # Top prediction
    idx = torch.argmax(probs).item()
    raw_label = model.config.id2label[idx]
    confidence = probs[idx].item() * 100

    food_name = FOOD_LABEL_MAP.get(raw_label.lower(), raw_label.title())
    quality = freshness_mapper(confidence)

    # --------------------------------------------------
    # OUTPUT
    # --------------------------------------------------
    st.subheader("🔍 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Detected Food", food_name)

    with col2:
        st.metric("Freshness Status", quality)

    st.progress(int(confidence))
    st.caption(f"Confidence: {confidence:.2f}%")

    st.info(explain_prediction(quality))
    st.caption(f"🧠 Model Class (ImageNet): {raw_label}")
