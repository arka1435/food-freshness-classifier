# 🍎 AI-Powered Food Freshness Assessment System

An end-to-end computer vision application that analyzes food images and estimates
their freshness using deep learning–based vision models.  
The system supports both **image upload** and **live camera input** for real-time food quality assessment.

Designed for placement and internship evaluation to demonstrate practical AI/ML deployment skills.

---

## 📌 Problem Statement
Food spoilage causes health risks and unnecessary waste.  
Manual visual inspection is subjective and error-prone.  
This project aims to assist users by automatically assessing food freshness
from images using computer vision techniques.

---

## 🚀 Solution Overview
The system allows users to:
- Upload a food image **or**
- Capture a food image using a **live camera**

It then provides:
- Detected food category (user-friendly)
- Freshness status (Fresh / Okay / Avoid)
- Confidence score
- Top-k prediction insights
- Explainable outputs for transparency

The application runs as an interactive web app using Streamlit.

---

## 🧠 Model & Approach
- **Model**: Vision Transformer (ViT)
- **Pre-training**: ImageNet
- **Frameworks**: PyTorch, Hugging Face Transformers

### Why Vision Transformer?
- Captures global visual context effectively
- Industry-relevant deep learning architecture
- Performs well on complex image classification tasks

---

## 🏗️ System Architecture
1. User provides input (image upload or live camera capture)
2. Image is resized and normalized
3. Vision Transformer extracts visual features
4. Softmax layer computes class probabilities
5. Confidence-based logic determines freshness level
6. Results are displayed in real time via web UI

---

## ✨ Key Features
- 📷 Image upload support
- 🎥 Live camera-based food image capture
- 🍏 User-friendly food category mapping
- 📊 Confidence score visualization
- 📈 Top-3 prediction analysis
- 🧠 Explainable decision output
- 🌐 Clean and interactive Streamlit interface

---

## ▶️ How to Run Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/arka1435/food-freshness-classifier.git
cd food-freshness-classifier
