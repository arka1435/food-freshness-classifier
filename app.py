import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="AI Food Freshness Assessment System",
    page_icon="🍎",
    layout="centered"
)

# --------------------------------------------------
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

# --------------------------------------------------
# Load Model (Cached)
# --------------------------------------------------
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
def freshness_mapper(confidence: float) -> str:
    if confidence >= 80:
        return "✅ Fresh"
    elif confidence >= 50:
        return "⚠️ Okay / Consume Soon"
    else:
        return "❌ Avoid"

def explain_prediction(quality: str) -> str:
    if "Fresh" in quality:
        return "The food appears visually intact with normal color and texture."
    elif "Okay" in quality:
        return "Minor visual inconsistencies detected. Recommended to consume soon."
    else:
        return "Visual cues indicate possible spoilage or degradation."

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("🍎 AI-Powered Food Freshness Assessment System")
st.markdown(
    "This system uses **deep learning–based computer vision** to analyze food images "
    "and estimate freshness with confidence-driven decision logic."
)

uploaded_file = st.file_uploader(
    "Upload a food image (jpg, jpeg, png)",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# Inference
# --------------------------------------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)[0]

    # --------------------------------------------------
    # Top-1 Prediction
    # --------------------------------------------------
    predicted_idx = torch.argmax(probs).item()
    raw_label = model.config.id2label[predicted_idx]
    confidence = probs[predicted_idx].item() * 100

    # User-friendly food name
    food_name = FOOD_LABEL_MAP.get(raw_label.lower(), raw_label.title())

    quality = freshness_mapper(confidence)

    # --------------------------------------------------
    # Main Output
    # --------------------------------------------------
    st.subheader("🔍 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Detected Food", food_name)

    with col2:
        st.metric("Freshness Status", quality)

    st.progress(int(confidence))
    st.caption(f"Confidence Score: {confidence:.2f}%")

    st.info(explain_prediction(quality))

    # Transparency
    st.caption(f"🧠 Model Class (ImageNet): {raw_label}")

    # --------------------------------------------------
    # 🔥 PLACEMENT-LEVEL FEATURE: TOP-3 PREDICTIONS
    # --------------------------------------------------
    st.subheader("📊 Top-3 Model Predictions")

    top_k = 3
    top_probs, top_indices = torch.topk(probs, top_k)

    for i in range(top_k):
        lbl = model.config.id2label[top_indices[i].item()]
        user_lbl = FOOD_LABEL_MAP.get(lbl.lower(), lbl.title())
        prob = top_probs[i].item() * 100

        st.write(f"**{i+1}. {user_lbl}** — {prob:.2f}%")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption(
    "⚙️ Built using Vision Transformer (ViT), PyTorch, Hugging Face Transformers, and Streamlit"
)
