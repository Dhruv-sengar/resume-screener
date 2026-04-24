import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def main():
    """
    Main function to execute the ML pipeline:
    1. Load and clean the dataset
    2. Split into train/test sets
    3. Vectorize text using TF-IDF
    4. Train a Logistic Regression model
    5. Evaluate and save the artifacts
    """
    print("🚀 Starting Resume Screening Model Training Pipeline...\n")

    # 1. Define relative paths
    dataset_path = "resume_ml/dataset.csv"
    model_path = "resume_ml/model.pkl"
    vectorizer_path = "resume_ml/vectorizer.pkl"

    # 2. Load the dataset
    print(f"📂 Loading dataset from '{dataset_path}'...")
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset not found at '{dataset_path}'")
        print("Please ensure you are running this script from the root directory.")
        return

    df = pd.read_csv(dataset_path)
    
    # 3. Clean the data
    initial_shape = df.shape
    
    # Drop rows that are literally NaN
    df.dropna(subset=['Category', 'Resume'], inplace=True)
    
    # Convert to string and strip whitespace
    df['Category'] = df['Category'].astype(str).str.strip()
    df['Resume'] = df['Resume'].astype(str).str.strip()
    
    # Drop rows where the string is empty after stripping
    df = df[(df['Category'] != '') & (df['Resume'] != '')]
    
    final_shape = df.shape
    print(f"🧹 Data cleaned. Removed {initial_shape[0] - final_shape[0]} invalid/empty rows.")
    print(f"📊 Total valid records: {final_shape[0]}\n")

    # 4. Separate features (X) and target labels (y)
    X = df['Resume']
    y = df['Category']

    # 5. Split into training and testing sets
    # Using stratify=y to ensure the train/test splits have the same proportion of classes
    print("✂️  Splitting data into training (80%) and testing (20%) sets (Stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 6. Initialize and fit the TF-IDF Vectorizer
    # TF-IDF converts text into numerical features based on term frequency and inverse document frequency
    print("🔤 Transforming text data using TF-IDF Vectorization...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=8000,
        ngram_range=(1, 2)
    )
    
    # Fit the vectorizer only on training data to prevent data leakage, then transform
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # 7. Initialize and train the Logistic Regression Model
    # multi_class="multinomial" is robust for multi-class classification
    # max_iter is set high to ensure convergence on large datasets
    print("🤖 Training Logistic Regression Classifier...")
    model = LogisticRegression(
        max_iter=2000,
        multi_class="multinomial",
        random_state=42
    )
    
    model.fit(X_train_tfidf, y_train)
    print("✅ Model training complete.\n")

    # 8. Evaluate the model
    print("📈 Evaluating Model Performance...")
    y_pred = model.predict(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Accuracy: {accuracy * 100:.2f}%\n")
    
    print("📋 Classification Report:")
    print("-" * 60)
    print(classification_report(y_test, y_pred))
    print("-" * 60)

    # 9. Save the Model and Vectorizer Artifacts
    print("\n💾 Saving trained artifacts...")
    
    # Ensure the directory exists (in case the relative path structure expects it)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"✔️  Model saved to '{model_path}'")

    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"✔️  Vectorizer saved to '{vectorizer_path}'")

    print("\n🎉 Pipeline finished successfully. The model is ready for the Flask API.")


if __name__ == "__main__":
    main()
