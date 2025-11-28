import pandas as pd

try:
    df = pd.read_csv('data/davidson_labeled_data.csv')
    print("Columns:", df.columns.tolist())
    print("First 5 rows:")
    print(df.head())
    print("\nClass distribution:")
    print(df['class'].value_counts())
except Exception as e:
    print(f"Error reading file: {e}")
