# 🍎 AI-Powered Food Freshness Assessment System

An end-to-end computer vision application that analyzes food images and estimates
their freshness using deep learning–based vision models.  
Designed to demonstrate real-world application of AI/ML concepts for placement and internship evaluation.

---

## 📌 Problem Statement
Food spoilage leads to health risks and unnecessary food waste.  
Consumers often rely on visual inspection, which can be subjective and unreliable.
This project aims to assist users by automatically assessing food freshness from images
using computer vision techniques.

---

## 🚀 Solution Overview
This system allows users to upload food images and receive:
- Detected food category (user-friendly)
- Freshness status (Fresh / Okay / Avoid)
- Confidence score
- Top-k prediction analysis
- Explainable output for better trust

The application is deployed as an interactive web interface using Streamlit.

---

## 🧠 Model & Approach
- **Model**: Vision Transformer (ViT)
- **Pre-training**: ImageNet
- **Framework**: PyTorch + Hugging Face Transformers

### Why Vision Transformer?
- Captures global image context effectively
- Industry-relevant deep learning architecture
- Strong performance on image classification tasks

---

## 🏗️ System Architecture
1. User uploads a food image
2. Image is resized and normalized
3. Vision Transformer extracts visual features
4. Softmax layer computes class probabilities
5. Confidence-driven logic determines freshness status
6. Results are displayed via web UI

---

## ✨ Key Features
- 📷 Image upload and real-time inference
- 🍏 User-friendly food category mapping
- 📊 Confidence score visualization
- 🔍 Top-3 prediction analysis
- 🧠 Explainability for model decisions
- 🌐 Clean and interactive web interface

---

## ▶️ How to Run Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/arka1435/food-freshness-classifier.git
cd food-freshness-classifier
