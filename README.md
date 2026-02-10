# Text Classification using TensorFlow

This project implements a **sentiment analysis system** using TensorFlow and TensorFlow Hub to classify IMDB movie reviews as positive or negative.

---

## 📌 Features

- Binary sentiment classification
- Pre-trained word embeddings from TensorFlow Hub
- Simple and efficient neural network
- Uses IMDB Reviews dataset
- End-to-end training and evaluation pipeline

---

## 🧠 Tech Stack

- Python
- TensorFlow / Keras
- TensorFlow Hub
- TensorFlow Datasets
- NumPy

---

## 📂 Project Structure
.
├── Text Classification with Tensorflow.py
├── README.md
└── .gitignore


---

## 📊 Dataset

- **IMDB Reviews Dataset**
- Loaded using `tensorflow_datasets`
- 50,000 movie reviews
- Labels:
  - 0 → Negative
  - 1 → Positive

---

## 🧠 Model Architecture

- Pre-trained text embedding layer (TensorFlow Hub)
- Dense layer with ReLU activation
- Output layer for binary classification

Loss Function:
- Binary Crossentropy (from logits)

Optimizer:
- Adam

---

## 🚀 How to Run

### 1️⃣ Install dependencies
```bash
pip install tensorflow tensorflow-datasets tensorflow-hub numpy

2️⃣ Run the script
python "Text Classification with Tensorflow.py"
