---
title: Cat Dog Classifier
emoji: 🔥
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 5.20.1
app_file: app.py
pinned: false
license: mit
short_description: '🐶🐱 Cat vs. Dog Classifier – AI-Powered Pet Recognition '
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference


# 🐱🐶 Cat vs Dog Classifier - Deep Learning Image Classification

[![Hugging Face Spaces](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-blue?logo=huggingface)](https://huggingface.co/spaces/dkg-2/cat_dog_classifier)

A lightweight deep learning classifier to distinguish between **cats** and **dogs** using transfer learning. Built using TensorFlow and deployed with Gradio on Hugging Face Spaces.

---

## 🧠 Model Overview

This image classification model uses a custom CNN, fine-tuned on thousands of labeled images from the Dogs vs Cats Kaggle dataset.

---

## 📦 Dataset

- **Source**: [Kaggle - Dogs vs. Cats Dataset](https://www.kaggle.com/c/dogs-vs-cats/data)
- **Description**: 25,000 high-quality labeled images (cats and dogs)

---


## 🛠 Tech Stack

- 🔹 **Framework**: TensorFlow / Keras
- 🔹 **Image Processing**: Pillow, OpenCV
- 🔹 **Deployment**: Gradio + Hugging Face Spaces

---

## 🧪 Live Demo

👉 [Try the Classifier on Hugging Face](https://huggingface.co/spaces/dkg-2/cat_dog_classifier)

Upload a cat or dog image and the model will tell you which one it is — with confidence.

---

## 💻 Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Cat-vs-Dog-Classifier.git
cd Cat-vs-Dog-Classifier
pip install -r requirements.txt
python app/app.py
