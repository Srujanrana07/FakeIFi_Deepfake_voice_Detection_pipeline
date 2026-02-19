# 🎙️ Deepfake Voice Detection  
**An End-to-End Audio Forensics & Detection Pipeline**

---

## 📌 Overview

This repository contains a research-driven, end-to-end **Deepfake Voice Detection** system that integrates audio signal processing, deep learning, and backend deployment to identify whether a voice sample is **authentic or synthetically generated**.

The project is designed as a complete ML pipeline, extending from raw audio ingestion to confidence-based predictions.  
A fully functional frontend application has also been developed and is available **upon request**.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Dataset](#-dataset)
- [Audio Processing Pipeline](#-audio-processing-pipeline)
- [Modeling Approach](#-modeling-approach)
- [Backend Service](#-backend-service)
- [Frontend Availability](#-frontend-availability)
- [Results & Performance](#-results--performance)
- [Limitations](#-limitations)
- [Future Enhancements](#-future-enhancements)
- [Repository Structure](#-repository-structure)
- [Installation & Usage](#-installation--usage)
- [Contact](#-contact)

---

## ✨ Key Features

- CNN-based deepfake voice detection  
- MFCC-driven audio representation  
- End-to-end ML inference pipeline  
- Audio preprocessing and feature extraction  
- Confidence-based binary classification  
- Backend API for model serving  
- Frontend application (private / on request)  

---

## 🏗️ System Architecture

```
Audio Input
  ↓
Preprocessing (Librosa)
  ↓
MFCC Feature Extraction
  ↓
CNN-Based Classifier
  ↓
Prediction + Confidence Score
  ↓
Backend API (Flask)
  ↓
Frontend Application (Private)
```

This architecture bridges **research-grade modeling** with **deployment-ready system design**.

---

## 📂 Dataset

- **Dataset:** SceneFake  
- **Sampling Rate:** 16 kHz, 16-bit mono WAV  

### Classes
- Real (bona fide speech)  
- Fake (synthetic / manipulated speech)  

### Dataset Split

| Split | Files |
|------|-------|
| Train | 13,185 |
| Dev | 12,843 |
| Eval | 32,746 |

Class imbalance in the training set is addressed using **SMOTE** during model training.

---

## 🎛️ Audio Processing Pipeline

Each audio sample undergoes:

- Loading and resampling at 16 kHz  
- Mono conversion  
- Silence trimming (optional)  
- Amplitude normalization  
- MFCC feature extraction  

This pipeline ensures consistency and robustness across varying audio conditions.

---

## 🤖 Modeling Approach

### CNN-Based Deepfake Classifier

- **Input:** MFCC feature maps  
- **Architecture:**
  - 1D Convolution layers with ReLU activation  
  - Dropout regularization  
  - MaxPooling  
  - Softmax output (Real vs Fake)  
- **Optimizer:** Adam  
- **Loss Function:** Sparse Categorical Cross-Entropy  

The deployed model achieves **~80–85% accuracy** on unseen evaluation data.

> Classical ML baselines (Random Forest, KNN) were explored during experimentation but are not part of the deployed inference pipeline.

---

## ⚙️ Backend Service

- **Framework:** Flask  

### Responsibilities
- Audio ingestion  
- Feature extraction  
- Model inference  
- Response generation  

The backend is designed to be **frontend-agnostic**, allowing easy integration with web or mobile clients.

---

## 🖥️ Frontend Availability

A fully implemented frontend application has been developed to complement this system, featuring:

- Step-based user flow  
- Audio recording and upload  
- Audio forensics visualizations  
- Prediction summaries with confidence scores  

🔒 **The frontend code is currently private.**

📩 If you are interested in:
- Reviewing the frontend  
- Collaborating on the project  
- Requesting a demo  

Please reach out via email.

---

## 📊 Results & Performance

- Binary classification: **Real vs Synthetic**  
- Evaluation accuracy: **~80–85%**  
- Confidence-based prediction outputs  
- Stable inference suitable for real-time usage  

---

## ⚠️ Limitations

- Evaluation limited to a single benchmark dataset  
- Generalization to unseen datasets not yet validated  
- Some overfitting observed in CNN training  
- No explicit adversarial defense mechanisms  

---

## 🚀 Future Enhancements

- 2D CNN / CRNN architectures on spectrograms  
- Transformer-based audio encoders (e.g., wav2vec 2.0)  
- Cross-dataset generalization experiments  
- Threshold calibration for production deployment  
- Public demo deployment  

---

## 📁 Repository Structure

```
srujanrana07-fakeifi_deepfake_voice_detection_pipeline/
    ├── app.py
    ├── LICENSE
    ├── requirements.txt
    ├── voice_analysis.py
    ├── voice_model.py
    └── models/
        └── label_encoder.joblib

```

---

## ⚙️ Installation & Usage

```bash
# Clone the repository
https://github.com/Srujanrana07/FakeIFi_Deepfake_voice_Detection_pipeline.git

# Install dependencies
pip install -r requirements.txt

# Run backend
python app.py
```

---

## 📬 Contact

📧 **Email:** srujanrana204@gmail.com  

For frontend access, collaboration, or discussion.
