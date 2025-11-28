import pandas as pd
import numpy as np
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

def clean_text(text):
    """
    Basic text cleaning: remove URLs, handles, special chars.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+', '', text) # Remove URLs
    text = re.sub(r'@\w+', '', text)    # Remove mentions
    text = re.sub(r'#\w+', '', text)    # Remove hashtags (optional, sometimes useful)
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    text = text.lower().strip()
    return text

def train_baseline(data_path):
    print(f"Checking for data at {data_path}...")
    
    if not os.path.exists(data_path):
        print("\n[ERROR] Hydrated data file not found!")
        print(f"Expected file: {data_path}")
        print("Please run 'scripts/hydrate_tweets.py' with valid Twitter API credentials first.")
        print("Cannot proceed with training without tweet text.")
        return

    print("Loading data...")
    df = pd.read_csv(data_path)
    
    if 'text' not in df.columns or 'Label' not in df.columns:
        # Try to infer columns if names are different
        # NAACL file usually has 'Label' or similar. Hydrated file will have 'text'.
        print(f"Columns found: {df.columns}")
        # Adjust column names if needed based on your hydration script output
        # Assuming 'text' is the tweet text and the label column is the second one from original csv
        pass

    # Drop rows with missing text
    df = df.dropna(subset=['text'])
    
    print("Preprocessing text...")
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Labels
    # The NAACL dataset has labels like 'racism', 'sexism', 'neither' (implied? or just 'none'?)
    # Let's check the label column. In NAACL_SRW_2016.csv it was the second column.
    # In the merged file, it should be preserved.
    # Let's assume the column name is 'Label' or similar.
    label_col = [c for c in df.columns if 'label' in c.lower() or 'class' in c.lower()]
    if label_col:
        y = df[label_col[0]]
    else:
        # Fallback: assume it's the column from the original merge that isn't ID or text
        # This is risky, so let's hope the merge preserved the name.
        # In NAACL_SRW_2016.csv, the header was missing in the file I read?
        # The file I read: 572342978255048705,racism
        # So it has no header. The hydration script might have assigned one or used 0/1.
        # If hydration script used `pd.read_csv(header=None)`, columns are 0, 1.
        # If so, 1 is label.
        if '1' in df.columns:
            y = df['1']
        elif 1 in df.columns:
            y = df[1]
        else:
            # Try to find a column with 'racism'/'sexism' values
            for col in df.columns:
                if df[col].astype(str).str.contains('racism').any():
                    y = df[col]
                    break
            else:
                print("Could not identify label column.")
                return

    X = df['clean_text']
    
    print(f"Training on {len(df)} samples.")
    print(f"Label distribution:\n{y.value_counts()}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))
    ])
    
    print("\nTraining Logistic Regression Baseline...")
    pipeline.fit(X_train, y_train)
    
    print("\nEvaluating...")
    y_pred = pipeline.predict(X_test)
    
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

if __name__ == "__main__":
    # Default path assumes running from scripts/ or root
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'hydrated_tweets.csv')
    train_baseline(data_path)
