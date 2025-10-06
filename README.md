# 🏠 House Price Prediction & Classification

## 🎯 Objective
The objective of this project is to:
- Predict house prices using **Linear Regression**
- Classify house prices into **Low**, **Medium**, and **High** using **K-Nearest Neighbors (KNN)**

---

## 📂 Dataset
- **File Name**: `house_Prediction_Data_Set.csv`
- **Features**: Multiple numerical features (e.g., area, bedrooms, etc.)
- **Target**: Final house price

### 🔧 Preprocessing Includes:
- Filling missing values using **median**
- Normalizing numerical features using **StandardScaler**
- Splitting dataset into **80% train** and **20% test**

---

## 🧠 ML Models Used

### 1. **Linear Regression** (for price prediction)
- Trained to predict continuous values of house prices.

### 2. **K-Nearest Neighbors Classifier** (for price category)
- Target categorized into:
  - **Low**: 0 to 15
  - **Medium**: 15 to 25
  - **High**: above 25

---

## 📊 Model Performance

### 🔹 Linear Regression:
| Metric                 | Score    |
|------------------------|----------|
| Mean Squared Error (MSE) | `24.29` |
| R-squared (R² Score)     | `0.67`  |  

### 🔹 KNN Classifier:
| Metric         | Score    |
|----------------|----------|
| Accuracy       | `0.86`   |  

📌 Also includes:
- **Confusion Matrix**
- **Classification Report** (Precision, Recall, F1-score)

---

## 📁 Project Structure

Folder: data/
    Contains your original and preprocessed CSV files:
    house_Prediction_Data_Set.csv

Folder: models/
    Stores your saved machine learning models:
    linear_regression_model.pkl
    final_knn_model.pkl

Folder: notebooks/
    Contains your Jupyter notebook:
    PROJ.ipynb

Folder: docs/
    Contains the PDF documentation of the project:
    Project Documentation.pdf

Folder: src/
    Contains Python scripts for preprocessing and training:
    preprocess.py
    train_models.py

File: README.md
    Provides an overview and instructions for the project.

File: requirements.txt
    Lists all required Python libraries for the project.

File: LICENSE
    Contains the open-source license (e.g., MIT License).



## 💾 Saved Models
- `models/linear_regression_model.pkl`
- `models/final_knn_model.pkl`

Load with:
```python
import joblib
lr_model = joblib.load('models/linear_regression_model.pkl')
knn_model = joblib.load('models/final_knn_model.pkl')
