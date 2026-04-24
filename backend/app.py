import os
import pickle
import PyPDF2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- Configuration & ML Artifact Paths ---
# Dynamically determine the root directory to handle varying execution contexts
# This ensures paths work perfectly when running "python backend/app.py" from the root.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE_DIR, 'resume_ml', 'model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'resume_ml', 'vectorizer.pkl')

print("🚀 Starting Intelligent Resume Screening API...\n")

# --- Load ML Artifacts ---
# Declared globally so they persist across standard incoming requests
model = None
vectorizer = None

try:
    print(f"📂 Loading model from: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
        
    print(f"📂 Loading vectorizer from: {VECTORIZER_PATH}")
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
        
    print("✅ Model and vectorizer successfully loaded into memory!\n")
except Exception as e:
    print(f"❌ Critical Error: Failed to load ML artifacts. Details: {str(e)}")
    print("Please ensure you have run the 'resume_ml/train_model.py' script first.")




# Handle OPTIONS requests explicitly if needed for pre-flight checking
@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for predicting resume job categories using the trained NLP model.
    Expected JSON Input:
    {
        "resume_text": "Your textual resume content here..."
    }
    """

    # 1. State Check: Ensure artifacts successfully loaded during startup
    if model is None or vectorizer is None:
        return jsonify({
            "error": "The ML model artifacts are not loaded. Check server logs."
        }), 500

    # 2. Basic Validation: Is the request a valid JSON payload?
    if not request.is_json:
        return jsonify({
            "error": "Invalid request format. The 'Content-Type' must be 'application/json'."
        }), 400
        
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Failed to parse JSON payload."}), 400
        
    resume_text = data.get('resume_text')
    
    # 3. Data Validation: Ensure resume_text is present, is a string, and isn't empty
    if resume_text is None:
        return jsonify({"error": "Missing 'resume_text' in the request body."}), 400
        
    if not isinstance(resume_text, str) or not resume_text.strip():
        return jsonify({"error": "The 'resume_text' field cannot be empty."}), 400

    # 4. Feature Engineering: Process the raw text into a numerical TF-IDF matrix
    try:
        # .transform() takes an iterable of strings, so we pass [resume_text]
        text_features = vectorizer.transform([resume_text])
    except Exception as e:
        return jsonify({"error": f"Error during text vectorization: {str(e)}"}), 500

    # 5. Prediction: Leverage Logistic Regression model
    try:
        predicted_probs = model.predict_proba(text_features)[0]
        
        # Get top 4 predictions to use 1 as primary and 3 as alternatives
        top_indices = np.argsort(predicted_probs)[::-1][:4]
        top_roles = [model.classes_[i] for i in top_indices]
        top_confidences = [round(float(predicted_probs[i]), 2) for i in top_indices]
        
        suggested_roles = [{"role": str(r), "confidence": c} for r, c in zip(top_roles[1:], top_confidences[1:])]
        
        # 6. Response: Format securely and cast numpy types to built-in Python types
        response = {
            "predicted_category": str(top_roles[0]),
            "confidence": top_confidences[0],
            "suggested_roles": suggested_roles
        }
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({"error": f"Error during model prediction logic: {str(e)}"}), 500



@app.route('/predict-pdf', methods=['POST'])
def predict_pdf():
    """
    Endpoint for predicting resume job categories using a PDF file.
    Expected Input: Multipart form data with a 'file' key containing the PDF.
    """

    if model is None or vectorizer is None:
        return jsonify({"error": "The ML model artifacts are not loaded."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Invalid file type. Only PDFs are allowed."}), 400

    try:
        # Read the PDF file directly from the in-memory FileStorage object
        pdf_reader = PyPDF2.PdfReader(file)
        extracted_text = ""
        
        # Iterate through all pages to extract text
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + " "
                
        cleaned_text = extracted_text.strip()
        
        if not cleaned_text:
            return jsonify({"error": "The PDF does not contain any extractable text."}), 400

    except Exception as e:
        return jsonify({"error": f"Error parsing PDF file: {str(e)}"}), 500

    # Feature Engineering
    try:
        text_features = vectorizer.transform([cleaned_text])
    except Exception as e:
        return jsonify({"error": f"Error during text vectorization: {str(e)}"}), 500

    # Prediction
    try:
        predicted_probs = model.predict_proba(text_features)[0]
        
        top_indices = np.argsort(predicted_probs)[::-1][:4]
        top_roles = [model.classes_[i] for i in top_indices]
        top_confidences = [round(float(predicted_probs[i]), 2) for i in top_indices]
        
        suggested_roles = [{"role": str(r), "confidence": c} for r, c in zip(top_roles[1:], top_confidences[1:])]
        
        response = {
            "predicted_category": str(top_roles[0]),
            "confidence": top_confidences[0],
            "suggested_roles": suggested_roles
        }
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": f"Error during model prediction: {str(e)}"}), 500


@app.route('/', methods=['GET'])
def index():
    """Basic health check API returning the state of the service."""
    return jsonify({
        "status": "Running",
        "service": "Intelligent Resume Screening API",
        "endpoints": ["POST /predict"],
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None
    }), 200


if __name__ == '__main__':
    # Start the robust Flask development loop on localhost at port 5000
    app.run(host='127.0.0.1', port=5000, debug=True)
