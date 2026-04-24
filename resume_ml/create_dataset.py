import os
import PyPDF2
import pandas as pd
import re

# Define relative paths
# This script is located in resume_ml/
# The dataset is located in ../dataset/ relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, '..', 'dataset')
OUTPUT_CSV = os.path.join(SCRIPT_DIR, 'dataset.csv')

def clean_text(text):
    """
    Cleans the extracted text by removing newlines, tabs, and extra spaces.
    Ensures the text fits on a single line for CSV compatibility.
    """
    if not text:
        return ""
    # Replace newlines, tabs, and multiple spaces with a single space
    cleaned = re.sub(r'\s+', ' ', text).strip()
    return cleaned

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a single PDF file.
    Returns the extracted text or None if extraction fails or text is empty.
    """
    text = ""
    try:
        with open(pdf_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            # Iterate through all pages
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
    except Exception as e:
        print(f"[ERROR] Could not read {os.path.basename(pdf_path)}: {e}")
        return None

    return clean_text(text)

def create_dataset():
    """
    Iterates through dataset folders, extracts PDF text, cleans it, and saves to CSV.
    """
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory not found at {DATASET_DIR}")
        return

    print(f"Scanning for resumes in: {os.path.abspath(DATASET_DIR)}\n")
    
    data = []
    processed_count = 0
    failed_count = 0
    
    # Get all subdirectories (categories)
    categories = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    
    for category in categories:
        category_path = os.path.join(DATASET_DIR, category)
        print(f"📂 Processing Category: {category}")
        
        files = os.listdir(category_path)
        for file_name in files:
            if file_name.lower().endswith('.pdf'):
                pdf_path = os.path.join(category_path, file_name)
                
                # Extract and clean text
                text = extract_text_from_pdf(pdf_path)
                
                if text:
                    data.append({'Category': category, 'Resume': text})
                    processed_count += 1
                else:
                    print(f"  ⚠️ No text found in: {file_name}")
                    failed_count += 1

    # Create DataFrame and save to CSV
    # Using quoting=1 (QUOTE_ALL) or quoting=None (default, minimizes) 
    # Pandas defaults usually handle quoting correctly for commas in string.
    # explicit quoting helps ensure robustness.
    import csv
    if data:
        df = pd.DataFrame(data)
        # quoting=csv.QUOTE_NONNUMERIC quotes all non-numeric fields (strings)
        df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print("\n" + "="*50)
        print(f"🎉 Success! CSV created at: {OUTPUT_CSV}")
        print(f"Total Resumes Processed: {processed_count}")
        print(f"Files with No Text: {failed_count}")
        print("="*50)
    else:
        print("\n❌ No data extracted. Please check if PDFs contain selectable text.")

if __name__ == "__main__":
    create_dataset()
