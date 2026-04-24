# Intelligent Resume Screening System

A full-stack, ML-powered system designed to automatically categorize resumes into specific job roles using Natural Language Processing (NLP). 

## 📂 Project Structure

```text
resume_screening/
├── backend/
│   ├── app.py              # Flask REST API serving the ML model
│   └── requirements.txt    # Python dependencies for the backend
├── resume_ml/
│   ├── dataset.csv         # Clean dataset of categorized resumes
│   ├── train_model.py      # ML pipeline script to clean, vectorize, and train
│   ├── model.pkl           # Trained Logistic Regression classifier artifact
│   └── vectorizer.pkl      # Trained TF-IDF Vectorizer artifact
└── README.md               # Project documentation (this file)
```

## 🚀 Quick Start Guide

### 1. Install Dependencies
Make sure you have Python installed, then install the required backend libraries:
```bash
pip install -r backend/requirements.txt
```

### 2. Train the Model
Ensure the essential ML artifacts (`model.pkl` and `vectorizer.pkl`) exist. If not, manually train your model first:
*(This will dynamically process `dataset.csv` and generate the `.pkl` artifacts natively in the `resume_ml/` folder.)*
```bash
python resume_ml/train_model.py
```

### 3. Run the Flask API
Start up the lightweight Python backend HTTP web server. It configures automatically to listen on `http://127.0.0.1:5000`.
```bash
python backend/app.py
```

## 🔌 API Reference

### `POST /predict`
Securely categorizes a given block of resume text into an industrial job profile, supplying an approximate prediction confidence score.

**Request Header:**
`Content-Type: application/json`

**Request Body Example:**
```json
{
  "resume_text": "Experienced Python Software Engineer with strong skills in Flask, SQL, and Machine Learning. Building REST APIs and maintaining scalable data pipelines."
}
```

**Response Example:**
```json
{
  "confidence": 0.87,
  "predicted_category": "DATA SCIENCE"
}
```

## 🧠 Technical Engine Context
- **Language Core**: Python 3
- **Machine Learning**: `scikit-learn` & `pandas`
  - **Vectorization Strategy**: TF-IDF Matrix Extraction (n_grams: 1-2, bounded by `max_features`: 8000)
  - **Algorithm Class**: Logsitic Regression using Multinomial Stratified Splitting Architecture
- **API Backbone**: Extensible `Flask` Framework Endpoint
