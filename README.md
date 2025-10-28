# 🧠 Jet Physics Analysis Using Deep Learning and Machine Learning  
### HLS4ML LHC Jet Dataset Challenge  
**Team Name:** Wistoria | **Team ID:** T-0894216  

**Team Members:**  
- Hanima (25-314040)  
- J Janasthuthi (25-300633)  
- Bhagyawanth (25-730761)  
📅 October 16, 2025  

---

## 🚀 Motivation
Jets are streams of particles produced during high-energy collisions such as those in the Large Hadron Collider (LHC).  
They provide vital insights into the fundamental forces of nature and can reveal rare or exotic phenomena.  

By combining **deep learning** and **anomaly detection** methods, we aim to uncover patterns and rare processes that traditional analyses might overlook — helping physicists push the boundaries of known physics.

---

## 🎯 Objective
To design and evaluate machine learning models that can:
1. Classify jets into their physical origins (quark, gluon, top, W, Z).  
2. Compare the effectiveness of image-based and tabular-based learning.  
3. Use unsupervised learning for anomaly detection to identify unusual jets that may signal new physics.

---

## 📊 Dataset Overview
**Dataset:** HLS4ML LHC Jet Dataset  
**Sources:** CERN Open Data / HLS4ML Collaboration  

Each `.h5` file contains:
- `jetImage`: 100×100 calorimeter energy maps (image representation of jets)  
- `jets_data`: 53 numerical physics-derived features  
- `target`: Categorical jet labels (`j_g`, `j_q`, `j_t`, `j_w`, `j_z`)  
- `FeatureNames`: Description of jet variables  

**Structure:**
- Training folder: 61 files (~610,000 jets)  
- Validation folder: 27 files (~270,000 jets)  

---

## ⚙️ Project Workflow

### **Part 1: Data Preprocessing**
- **Feature Handling:**  
  Separated image and tabular features. Missing values imputed using **median strategy**, robust to outliers common in physics datasets.
- **Tabular Data Preparation (PCA):**  
  Performed Principal Component Analysis (PCA) after standardization.  
  - 12 components retained 95.43% of variance.  
  - PCA reduced noise and model complexity while preserving important physics information.  
- **Dataset Summary:**  
  - Records: 10,000 (for initial test)  
  - Features: 53 tabular, 100×100 images  
  - Classes: 5  

---

### **Part 2: Model Development**

#### 🧩 **2.1 CNN on Image Data**
- 3 Convolutional blocks (32, 64, 128 filters)  
- Batch Normalization + MaxPooling + Dropout  
- Activation: ReLU; Optimizer: Adam  
- Checkpoints and early stopping implemented  

**Training Results:**  
- Smooth convergence with minimal overfitting.  
- Final Validation Accuracy ≈ **81.33%**  

**Interpretation:**  
The CNN successfully learned spatial energy patterns in jets, distinguishing between different origins (e.g., quark vs gluon).

---

#### 📊 **2.2 Models on Tabular Data**
Baseline and PCA-reduced datasets were used to train **Random Forest (RF)** classifiers.

| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|--------|-----------|------------|--------|----------|----------|
| Baseline RF | **81.50%** | 0.81 | 0.81 | 0.81 | 0.95 |
| PCA RF | 77.88% | 0.77 | 0.77 | 0.77 | 0.94 |

**Analysis:**  
- PCA improved training speed but caused a slight performance drop.  
- Baseline RF best captured the relationships among raw features.  

---

### **Part 3: Model Comparison**

| Metric | Tabular (RF) | Image (CNN) |
|---------|---------------|-------------|
| Accuracy | **81.50%** | 77.88% |
| Precision | 0.81 | 0.77 |
| Recall | 0.81 | 0.77 |
| F1-Score | 0.81 | 0.77 |
| ROC AUC | **0.95** | 0.94 |

**Physics Perspective:**  
- Tabular features summarize derived quantities (mass, momentum, etc.) — efficient but limited.  
- CNN directly learns from low-level calorimeter images, capturing **fine spatial structures** that human-engineered features may miss.  
- This highlights the power of **deep learning** for high-energy physics.

---

### **Part 4: Anomaly Detection**

#### ⚡ CNN Autoencoder
- Trained only on "normal" jets to learn reconstruction patterns.  
- Used **Mean Squared Error (MSE)** as the anomaly score.  
- Threshold = mean + 2σ of reconstruction error = **0.2504**.  

**Results:**
- **2410 anomalies detected** in test data.  
- Top 5 anomalies visualized: high reconstruction errors and distorted images.

**Physics Insight:**  
Detected anomalies may correspond to:
- **New physics signals** (e.g., beyond-Standard-Model particles)  
- **Rare Standard Model events**  
- **Detector artifacts/noise**  

---

## 🧠 Key Insights

| Aspect | Insight |
|--------|----------|
| **Feature Handling** | Median imputation preserves data integrity against outliers. |
| **PCA** | Reduced dimensions from 53 → 12 with 95% variance retained. |
| **CNN Classifier** | Learned spatial jet patterns directly from image energy distributions. |
| **Random Forest** | Provided strong baseline with high interpretability. |
| **Autoencoder** | Detected outlier jets using reconstruction error — a model-agnostic anomaly detection method. |

---

## 📈 Visual Outputs
| Plot | Description |
|------|--------------|
| **Scree Plot** | Explained variance vs principal component (PCA). |
| **CNN Accuracy/Loss** | Training convergence visualization. |
| **Confusion Matrices** | Class-wise accuracy for CNN and RF. |
| **ROC Curves** | Model comparison across classes. |
| **Reconstruction Errors** | Anomaly detection threshold and distribution. |

---

## 📦 Directory Structure
