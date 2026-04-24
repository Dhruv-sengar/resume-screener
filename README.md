<div align="center">

# 🤖 NexResume
### *AI-Powered Resume Intelligence Platform*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://reactjs.org)
[![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

**Instantly map optimal career trajectories from raw resume data using advanced ML models.**

[Live Demo](#) · [Report Bug](https://github.com/Dhruv-sengar/resume-screener/issues) · [Request Feature](https://github.com/Dhruv-sengar/resume-screener/issues)

</div>

---

## ✨ What is NexResume?

NexResume is a full-stack, AI-powered resume screening platform that analyzes any resume — either as a **PDF upload** or **raw text paste** — and instantly predicts the most suitable job roles using a trained Machine Learning pipeline.

Built with a modern **React 18** frontend and a **Flask** backend, NexResume delivers a premium SaaS-grade experience with real-time analysis, animated UI, dark/light mode support, and a full analysis history dashboard.

---

## 🚀 Key Features

| Feature | Description |
|---|---|
| 🧠 **ML Role Prediction** | Classifies resumes into 25+ career categories using TF-IDF + Logistic Regression |
| 📄 **PDF & Text Support** | Upload a PDF resume or paste raw text for instant analysis |
| 🎯 **Top Role Matching** | Displays the best matching role + 3 alternative career suggestions with confidence scores |
| 🗂️ **Analysis History** | Track and revisit all past resume scans from a sleek dashboard |
| 🌗 **Dark / Light Mode** | Premium BB-8 animated toggle for seamless theme switching |
| 🔐 **Auth Wall** | Email-based login/signup screen with animated staggered entry |
| 📊 **Skills Matrix** | Visual breakdown of detected skills from the resume |
| ⚡ **Auto-Scroll Results** | Smooth scroll to results immediately after analysis completes |
| 🎨 **Animated Background** | Floating ambient orbs with continuous physics animations |

---

## 🧱 Tech Stack

### Frontend
- **React 18** — Component-based UI
- **CSS Variables** — Full dark/light theme system
- **Custom CSS Animations** — Hover effects, staggered reveals, orb physics
- **Glassmorphism Design** — Premium frosted-glass cards and nav

### Backend
- **Flask** — Lightweight Python API server
- **Flask-CORS** — Cross-origin request handling
- **PyPDF2** — PDF text extraction
- **scikit-learn** — TF-IDF Vectorizer + Logistic Regression classifier
- **NumPy** — Probability ranking and array processing

### ML Pipeline
- **Dataset**: 2,500+ labeled resumes across 25 career categories
- **Vectorizer**: TF-IDF with N-gram support
- **Model**: Multinomial Logistic Regression with `predict_proba` for confidence scores
- **Output**: Top predicted role + 3 ranked alternatives with percentages

---

## 📁 Project Structure

```
resume-screener/
│
├── backend/
│   ├── app.py              # Flask API (predict + predict-pdf endpoints)
│   └── requirements.txt    # Python dependencies
│
├── frontend/
│   ├── public/
│   └── src/
│       ├── components/
│       │   ├── LoginSignup.jsx    # Auth screen
│       │   ├── ResumeForm.jsx     # Text input form
│       │   ├── ResumeUpload.jsx   # Drag-and-drop PDF upload
│       │   ├── ResultCard.jsx     # Analysis results display
│       │   ├── History.jsx        # Past scans dashboard
│       │   ├── Settings.jsx       # User settings panel
│       │   └── ThemeToggle.jsx    # BB-8 dark/light toggle
│       ├── services/
│       │   └── api.js             # API service layer
│       ├── App.js                 # Main app + routing state
│       └── App.css                # Global styles + animations
│
├── resume_ml/
│   ├── train_model.py      # ML training script
│   ├── dataset.csv         # Labeled resume dataset
│   └── generate_dataset.py # Dataset preparation utilities
│
└── README.md
```

---

## ⚙️ Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+
- npm

### 1. Clone the repository
```bash
git clone https://github.com/Dhruv-sengar/resume-screener.git
cd resume-screener
```

### 2. Train the ML Model
```bash
cd resume_ml
pip install scikit-learn pandas
python train_model.py
```
> This generates `model.pkl` and `vectorizer.pkl` in the root directory.

### 3. Start the Backend
```bash
cd backend
pip install -r requirements.txt
python app.py
```
> API runs at `http://127.0.0.1:5000`

### 4. Start the Frontend
```bash
cd frontend
npm install
npm start
```
> App runs at `http://localhost:3000`

---

## 🔌 API Reference

### `POST /predict`
Predict job category from raw resume text.

**Request:**
```json
{
  "resume_text": "Experienced Python developer with 5 years in ML..."
}
```

**Response:**
```json
{
  "predicted_category": "Machine Learning Engineer",
  "confidence": 0.92,
  "suggested_roles": [
    { "role": "Data Scientist", "confidence": 0.85 },
    { "role": "AI Researcher", "confidence": 0.78 },
    { "role": "Data Engineer", "confidence": 0.65 }
  ]
}
```

### `POST /predict-pdf`
Upload a PDF resume file (multipart/form-data).

| Field | Type | Description |
|---|---|---|
| `file` | `.pdf` | The resume PDF to analyze |

---

## 🗂️ Supported Job Categories

The model is trained to classify resumes across **25+ categories** including:

`Data Science` · `Machine Learning` · `Web Development` · `Backend Engineering` · `HR` · `Sales` · `Finance` · `Teaching` · `Healthcare` · `DevOps` · `Cybersecurity` · `Mobile Development` · `Product Management` · `UI/UX Design` · `Database Administration` · and more.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<div align="center">

Made with ❤️ by [Dhruv Sengar](https://github.com/Dhruv-sengar)

⭐ **Star this repo if you found it useful!**

</div>
