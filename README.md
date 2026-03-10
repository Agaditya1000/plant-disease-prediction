# 🌱 DEPLANT — Deep Learning for Plant Disease Prediction

## 📌 Project Overview
DEPLANT is an AI‑based plant disease detection system that uses deep learning models to identify plant diseases from leaf images. The system analyzes uploaded images, predicts the disease type, estimates infection severity, and provides results through a web interface.

The goal of this project is to support farmers with early disease detection and promote smart farming practices using Artificial Intelligence.

---

## 🚀 Features
- 🌿 Automatic plant disease detection from leaf images
- 🤖 Multiple ML/DL models implemented:
  - CNN (Convolutional Neural Network)
  - SVM (Support Vector Machine)
  - ResNet50 (Transfer Learning)
- 📊 Model comparison and performance evaluation
- 🔬 Disease severity estimation using HSV segmentation
- 🌐 Streamlit web interface for real‑time prediction
- 💬 AgriBot assistant for crop and disease guidance

---

## 📂 Dataset
This project uses the **PlantVillage Dataset**.

### Dataset Details
- Total Images: ~22,000
- Classes: 16 plant disease categories
- Crops included:
  - Tomato
  - Potato
  - Pepper

Dataset includes:
- Healthy leaves
- Diseased leaves

---

## ⚙️ Technologies Used

### Programming Language
- Python

### Libraries
- TensorFlow
- Keras
- OpenCV
- NumPy
- Scikit‑learn
- Matplotlib

### Deployment
- Streamlit

---

## 🧠 Models Implemented

### 1️⃣ CNN (Convolutional Neural Network)
- Baseline deep learning model
- Learns spatial patterns from images
- Training Accuracy: ~92%

---

### 2️⃣ SVM (Support Vector Machine)
- Traditional machine learning model
- Uses extracted features for classification
- Accuracy: ~63%

---

### 3️⃣ ResNet50 (Transfer Learning)
- Deep residual network
- Pretrained on ImageNet
- Training Accuracy: ~90.9%
- Validation Accuracy: ~87.3%
- Best performing model

---

## 🖼️ System Workflow

```
Leaf Image Input
       ↓
Image Preprocessing
       ↓
CNN / ResNet Model
       ↓
Disease Prediction
       ↓
HSV Segmentation
       ↓
Infected Area Percentage
       ↓
Disease Stage (Early / Moderate / Severe)
```

---

## 🔬 Severity Detection

Severity is calculated using image processing techniques.

### Steps
1. Convert image from RGB → HSV
2. Detect infected regions using color thresholds
3. Calculate infected pixels

### Formula

```
Severity (%) = (Infected Pixels / Total Pixels) × 100
```

### Stage Classification

| Severity | Stage |
|--------|--------|
| <10% | Early Stage |
| 10–30% | Moderate Stage |
| >30% | Severe Stage |

---

## 🌐 Web Application (Streamlit)

The trained model is deployed using **Streamlit**, allowing users to:

- Upload plant leaf images
- Get instant disease predictions
- View infection severity
- Access AgriBot agricultural assistant

---

## 📊 Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## 📈 Results Summary

| Model | Training Accuracy | Validation Accuracy |
|------|------|------|
| CNN | ~92% | ~63% |
| SVM | ~63% | Lower than CNN |
| ResNet50 | **~90.9%** | **~87.3%** |

ResNet50 achieved the best performance and was selected as the final model.

---

## 🔮 Future Improvements
- Mobile application integration
- Real‑time camera disease detection
- Larger real‑world datasets
- IoT‑based crop monitoring
- Automated treatment recommendations

---

## 👥 Contributors
- Aditya Kumar Gupta  
- Pratishtha Srivastava  
- Dr. Swathy R (Supervisor)

Department of Networking and Communications  
SRM Institute of Science and Technology  
Chennai, India

---

## 📜 License
This project is developed for academic and research purposes.
