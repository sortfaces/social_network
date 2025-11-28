import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix
import sys
import os

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    # The file appears to be tab-separated based on the preview
    try:
        df = pd.read_csv(filepath, sep='\t')
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    return df

def analyze_annotations(df):
    print("\n--- Data Overview ---")
    print(f"Total Tweets: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    
    # Extract Expert labels
    expert_labels = df['Expert']
    print("\nExpert Label Distribution:")
    print(expert_labels.value_counts())

    # Extract Amateur columns (assuming they start with 'Amateur_')
    amateur_cols = [c for c in df.columns if c.startswith('Amateur_')]
    print(f"\nNumber of Amateur Annotator Columns: {len(amateur_cols)}")

    # Calculate Amateur Majority Vote
    print("\nCalculating Amateur Majority Vote...")
    amateur_votes = df[amateur_cols]
    
    # Function to get majority vote, ignoring NaNs/empty strings
    def get_majority(row):
        # Filter out empty or NaN values
        valid_votes = [v for v in row if pd.notna(v) and str(v).strip() != '']
        if not valid_votes:
            return "No Vote"
        return max(set(valid_votes), key=valid_votes.count)

    amateur_majority = amateur_votes.apply(get_majority, axis=1)
    
    df['Amateur_Majority'] = amateur_majority
    
    # Filter out tweets where amateurs didn't vote (if any)
    valid_comparison = df[df['Amateur_Majority'] != "No Vote"].copy()
    
    print(f"\nTweets with valid Amateur votes: {len(valid_comparison)}")
    
    # Comparison: Expert vs Amateur Majority
    y_true = valid_comparison['Expert']
    y_pred = valid_comparison['Amateur_Majority']
    
    print("\n--- Agreement Analysis: Expert vs. Amateur Majority ---")
    
    # Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    print(f"Cohen's Kappa: {kappa:.4f}")
    
    # Classification Report
    print("\nClassification Report (Expert as Ground Truth):")
    print(classification_report(y_true, y_pred))
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    labels = sorted(list(set(y_true.unique()) | set(y_pred.unique())))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"True_{l}" for l in labels], columns=[f"Pred_{l}" for l in labels])
    print(cm_df)

    # Save results
    # Use the global file_path or pass it in
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, '..', 'results', 'annotator_analysis_results.csv')
    
    # Ensure results directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    valid_comparison[['TweetID', 'Expert', 'Amateur_Majority']].to_csv(output_path, index=False)
    print(f"\nDetailed comparison saved to {output_path}")

if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'NLP+CSS_2016.csv')
    
    # Check if file exists
    if not os.path.exists(file_path):
        # Fallback for the user's specific path structure if running from root
        file_path = os.path.join('e:\\hatepeech\\data\\NLP+CSS_2016.csv')
        
    if os.path.exists(file_path):
        df = load_data(file_path)
        if df is not None:
            analyze_annotations(df)
    else:
        print(f"File not found at {file_path}")
