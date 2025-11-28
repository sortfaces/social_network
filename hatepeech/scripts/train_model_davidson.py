import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import os

def train_davidson_model(file_path):
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    # Davidson dataset mapping: 0: Hate Speech, 1: Offensive Language, 2: Neither
    # For binary classification (Hate vs Not Hate), we could group 0 as Hate and 1+2 as Not Hate,
    # or keep it multi-class. Let's keep it multi-class as it's more informative.
    
    class_names = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
    df['label_name'] = df['class'].map(class_names)

    print(f"Dataset shape: {df.shape}")
    print("Class distribution:")
    print(df['label_name'].value_counts())

    # Split data
    X = df['tweet']
    y = df['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Build pipeline
    print("\nTraining Logistic Regression model (TF-IDF)...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    # Evaluate
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Hate Speech', 'Offensive Language', 'Neither']))

    return pipeline

if __name__ == "__main__":
    # Assuming the script is run from the root or scripts dir
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'davidson_labeled_data.csv')
    
    train_davidson_model(data_path)
