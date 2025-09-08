# 🧠 Stroke Detection Streamlit App

![App Interface](img/stroke.jpg)

This project was developed as my **final project during my internship at Aman Holding**.  
It is a **Streamlit web application** that predicts whether a person is at risk of stroke based on healthcare data, while also providing dataset visualizations, insights, and model explainability with **SHAP values**.

---

## 📊 Project Overview
- Uses a **healthcare dataset from Kaggle** for training and evaluation.
- Provides **interactive visualizations** to understand dataset patterns.
- A trained **Machine Learning model** is integrated to predict stroke likelihood.
- **SHAP values** are used to explain the model’s predictions in a transparent way.
- Includes a **Jupyter Notebook** documenting:
  - Data preprocessing pipeline  
  - Feature engineering  
  - Model selection and evaluation  

---

## 🚀 Features
- **Stroke Prediction**: Input patient health attributes and get a prediction.
- **Visual Analytics**: Explore dataset statistics and plots.
- **Explainable AI**: SHAP values show *why* the model made its prediction.
- **End-to-End Pipeline**: From raw data → preprocessing → model → deployment.

---

## 🛠️ Tech Stack
- **Python 3.8+**  
- **Libraries used**:  
  - `numpy`  
  - `pandas`  
  - `scikit-learn`  
  - `shap`  
  - `matplotlib`  
  - `seaborn`  
  - `plotly`  
  - `streamlit`  
  - `joblib`  
  - `os`, `PIL`  

---

## 📦 Installation

Clone the repository:
```bash
git clone https://github.com/Seif-Elmezaien/Stroke-Detection.git
cd Stroke-Detection
```


## ⬇️ Install dependencies
```bash
pip install -r requirements.txt
```


## ▶️Usage
```bash
streamlit run app.py
```

** This project is for educational purposes as part of my internship. The dataset is publicly available on Kaggle. **
