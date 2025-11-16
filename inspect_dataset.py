# inspect_dataset.py
import pandas as pd
import os

def inspect_enron_dataset():
    """Inspect your Enron dataset to understand its structure"""
    data_path = "data/enron_spam_data.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ Dataset not found at: {data_path}")
        return
    
    df = pd.read_csv(data_path)
    
    print("=" * 60)
    print("🔍 ENRON DATASET INSPECTION")
    print("=" * 60)
    
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📝 Columns: {list(df.columns)}")
    print()
    
    print("📋 Column details:")
    for col in df.columns:
        print(f"   {col}: {df[col].dtype} | Unique values: {df[col].nunique()} | Null values: {df[col].isnull().sum()}")
    
    print()
    print("📄 First 3 rows:")
    print(df.head(3))
    
    print()
    print("🔢 Statistical summary:")
    print(df.describe(include='all'))
    
    # Try to identify text and label columns
    print()
    print("🎯 Identifying text and label columns:")
    
    text_candidates = []
    label_candidates = []
    
    for col in df.columns:
        col_data = df[col]
        
        # Check for text columns (string type with reasonable length)
        if col_data.dtype == 'object':
            avg_length = col_data.str.len().mean()
            if avg_length > 20:  # Reasonable minimum length for email text
                text_candidates.append((col, avg_length))
        
        # Check for label columns (binary or few unique values)
        unique_vals = col_data.unique()
        if len(unique_vals) <= 5:
            label_candidates.append((col, unique_vals))
    
    print("   Text column candidates:")
    for col, avg_len in text_candidates:
        print(f"     '{col}' - avg length: {avg_len:.1f} chars")
    
    print("   Label column candidates:")
    for col, vals in label_candidates:
        print(f"     '{col}' - values: {vals}")
    
    # Show value counts for potential label columns
    print()
    print("📈 Value counts for potential label columns:")
    for col, _ in label_candidates:
        print(f"   {col}:")
        print(df[col].value_counts())
        print()

if __name__ == "__main__":
    inspect_enron_dataset()