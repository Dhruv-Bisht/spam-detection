# load_enron_data.py
import pandas as pd
import os

def load_actual_enron_data(enron_path="data/enron"):
    """
    Load actual Enron dataset from directory structure
    Adjust this function based on your Enron dataset structure
    """
    emails = []
    labels = []
    
    # Example structure - adjust based on your dataset
    try:
        # If you have spam/ham folders
        spam_path = os.path.join(enron_path, "spam")
        ham_path = os.path.join(enron_path, "ham")
        
        # Load spam emails
        if os.path.exists(spam_path):
            for filename in os.listdir(spam_path):
                with open(os.path.join(spam_path, filename), 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    emails.append(content)
                    labels.append(1)  # spam
        
        # Load ham emails
        if os.path.exists(ham_path):
            for filename in os.listdir(ham_path):
                with open(os.path.join(ham_path, filename), 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    emails.append(content)
                    labels.append(0)  # ham
                    
        df = pd.DataFrame({'text': emails, 'label': labels})
        print(f"✅ Loaded {len(df)} emails from Enron dataset")
        return df
        
    except Exception as e:
        print(f"❌ Error loading Enron data: {e}")
        return None

def save_enron_csv():
    """Convert Enron data to CSV format"""
    df = load_actual_enron_data()
    if df is not None:
        df.to_csv("data/enron_spam_data.csv", index=False)
        print("✅ Enron data saved to CSV")
    else:
        print("❌ Failed to load Enron data")

if __name__ == "__main__":
    save_enron_csv()