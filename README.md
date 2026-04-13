# 🌿 AI-Based Multi-Level Vegetation Analysis System

## 🎯 Project Overview

This project presents an **AI-driven system for vegetation analysis** using deep learning techniques at multiple levels:

* **Macro Level** → Satellite Image Classification
* **Micro Level** → Tree Detection & Vegetation Coverage Estimation

The system enables users to upload images and receive intelligent insights about land type, vegetation density, and environmental characteristics.

---

## 🧠 Core Objectives

* Classify land types from satellite imagery using CNN
* Detect individual trees from aerial images using object detection
* Estimate vegetation coverage percentage
* Provide a user-friendly interface for real-time analysis
* Build a scalable backend using FastAPI

---

## ⚙️ System Architecture

```
User Upload Image
        ↓
Select Mode (Satellite / Tree Detection)
        ↓
FastAPI Backend
        ↓
Deep Learning Model Inference
        ↓
Post-Processing
        ↓
Frontend Visualization
```

---

## 🌍 MODULE 1 — Satellite Image Classification

### 📌 Purpose

To classify land types from satellite images.

### 🧠 Model

* Convolutional Neural Network (CNN)
* Trained on EuroSAT Dataset

### 📥 Input

* Image resized to 224×224
* Normalized pixel values

### 📤 Output

* Land Type:

  * Forest
  * Residential
  * River
  * Agricultural
* Confidence Score

### 🌿 Vegetation Estimation

| Land Type    | Vegetation Coverage |
| ------------ | ------------------- |
| Forest       | 70–100%             |
| Agricultural | 40–70%              |
| Residential  | 10–40%              |

---

## 🌳 MODULE 2 — Tree Detection (Core Module)

### 📌 Purpose

To detect trees and analyze vegetation from aerial imagery.

### 🧠 Model

* DeepForest (Pretrained Object Detection Model)

### 📥 Input

* Aerial images (.jpg / .tif)

### 📤 Output

* Bounding Boxes around trees
* Tree Count
* Vegetation Level
* Coverage Percentage

---

## 🔧 Post-Processing Pipeline

* **Confidence Thresholding** → Remove detections < 0.25
* **Size Filtering** → Remove noise (small boxes)
* **Duplicate Removal** → Keep highest confidence detections

---

## 📊 Vegetation Metrics

### 🌲 Tree Count

Total detected trees

### 🌿 Vegetation Level

| Tree Count | Level    |
| ---------- | -------- |
| < 10       | Low      |
| 10–30      | Moderate |
| > 30       | High     |

### 📈 Coverage Percentage

Coverage is estimated using:

```
Coverage (%) = (Total Bounding Box Area / Image Area) × 100
```

⚠️ Note: This is an approximation (bounding boxes ≠ exact segmentation)

---

## 💻 Tech Stack

### 🔧 Backend

* FastAPI
* Python
* TensorFlow / Keras
* DeepForest

### 🌐 Frontend

* HTML
* CSS
* JavaScript

### 🧠 AI Concepts Used

* Convolutional Neural Networks (CNN)
* Transfer Learning
* Object Detection
* Bounding Boxes
* Confidence Scores
* Feature Extraction

---

## 🔌 API Endpoints

| Endpoint             | Description               |
| -------------------- | ------------------------- |
| `/predict-satellite` | Land classification       |
| `/predict-trees`     | Tree detection & analysis |

---

## 🖼️ Features

* Upload image for analysis
* Switch between modes (Satellite / Tree Detection)
* Visual output with bounding boxes
* Real-time vegetation insights

---

## 🔄 Workflow

1. User uploads image
2. Selects analysis type
3. Backend processes request
4. Model performs inference
5. Post-processing applied
6. Results returned
7. Frontend displays output

---

## ⚠️ Limitations

* Tree detection works best on aerial imagery
* Coverage estimation is approximate
* Performance depends on image quality

---

## ⚙️ Installation & Setup Guide

Follow these steps to run the project locally.

---

### 🔹 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

---

### 🔹 2. Backend Setup (FastAPI)

#### 📌 Step 1: Create Virtual Environment

```bash
python -m venv venv
```

#### 📌 Step 2: Activate Environment

**Windows:**

```bash
venv\Scripts\activate
```

**Mac/Linux:**

```bash
source venv/bin/activate
```

---

#### 📌 Step 3: Install Dependencies from the backend directory

```bash
pip install -r requirements.txt
```

> If requirements.txt is missing, install manually:

```bash
pip install fastapi uvicorn tensorflow numpy opencv-python pillow deepforest
```

---

#### 📌 Step 4: Run Backend Server

```bash
uvicorn main:app --reload
```

Server will start at:

```
http://127.0.0.1:8000
```

---

### 🔹 3. Frontend Setup

Navigate to frontend folder:

```bash
cd frontend
```

Run a local server:

```bash
python -m http.server 5500
```

Open browser:

```
http://localhost:5500
```

---

### 🔹 4. Using the Application

1. Open the frontend in browser
2. Upload an image
3. Select mode:

   * Satellite Classification
   * Tree Detection
4. View results:

   * Predictions
   * Vegetation metrics
   * Output image

---


## ⚠️ Common Issues & Fixes

---

### ❌ Port already in use

```bash
uvicorn main:app --reload --port 8001
```

---

### ❌ No detections / 0% coverage

* Use proper aerial images
* Ensure confidence threshold is not too high

---

## 💡 Pro Tips

* Always activate virtual environment before running backend
* Use good-quality images for better predictions
* Keep backend running while using frontend

---

## 🚀 Future Enhancements

* 🌱 Leaf-based Plant Species Detection (Upcoming Module)
* 🌍 Real-time satellite monitoring
* 📊 Integration with GIS systems
* 🎯 Semantic segmentation for precise coverage

---

## 💥 Final Output

### Satellite Mode

* Land Type
* Confidence Score
* Estimated Vegetation

### Tree Detection Mode

* Tree Count
* Vegetation Level
* Coverage %
* Annotated Image

---

## 📁 Project Structure

```
├── backend/
│   ├── main.py
│   ├── models/
│   ├── routes/
│   └── utils/
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── images/
├── outputs/
└── README.md
```

---

## 🧠 Key Takeaway

This project transforms traditional vegetation analysis into a **data-driven, AI-powered system** capable of extracting meaningful environmental insights from images.

---

## 👨‍💻 Author

* Your Name

---

## ⭐ If you found this useful, consider giving it a star!
