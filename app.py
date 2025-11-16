# app.py
from flask import Flask, render_template, request, jsonify
import os
import joblib
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

# --- Enron Dataset Spam Model ---
MODEL_PATH = "enron_spam_model.pkl"
DATA_PATH = "data/enron_spam_data.csv"  # Path to your Enron dataset

def inspect_dataset(df):
    """Inspect the dataset structure and identify columns"""
    print("🔍 Inspecting dataset structure...")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"First few rows:")
    print(df.head())
    print(f"Column info:")
    print(df.info())
    
    # Try to identify text and label columns
    text_columns = []
    label_columns = []
    
    for col in df.columns:
        # Check if column contains text data
        if df[col].dtype == 'object' and df[col].str.len().mean() > 10:
            text_columns.append(col)
        # Check if column contains binary labels (spam/ham)
        elif df[col].nunique() <= 3 and set(df[col].unique()).issubset({0, 1, '0', '1', 'spam', 'ham', 'Spam', 'Ham'}):
            label_columns.append(col)
    
    print(f"Potential text columns: {text_columns}")
    print(f"Potential label columns: {label_columns}")
    
    return text_columns, label_columns

def load_enron_data():
    """Load and prepare Enron spam dataset"""
    print("📁 Loading Enron dataset...")
    
    try:
        # If you have the Enron dataset as CSV
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            print(f"✅ Loaded Enron dataset with {len(df)} emails")
            
            # Inspect the dataset
            text_cols, label_cols = inspect_dataset(df)
            
            # Use the first text column and first label column found
            if text_cols and label_cols:
                text_col = text_cols[0]
                label_col = label_cols[0]
                print(f"📝 Using text column: '{text_col}', label column: '{label_col}'")
                
                # Clean the data
                df = df.dropna(subset=[text_col, label_col])
                df[text_col] = df[text_col].astype(str)
                
                # Convert labels to numeric if needed
                if df[label_col].dtype == 'object':
                    label_mapping = {'ham': 0, 'spam': 1, 'Ham': 0, 'Spam': 1}
                    df[label_col] = df[label_col].map(label_mapping).fillna(df[label_col])
                    df[label_col] = pd.to_numeric(df[label_col], errors='coerce')
                
                df = df.dropna(subset=[label_col])
                df[label_col] = df[label_col].astype(int)
                
                # Rename columns to standard names for easier processing
                df_standardized = df.rename(columns={text_col: 'text', label_col: 'label'})
                print(f"📊 Final dataset: {len(df_standardized)} emails, {df_standardized['label'].sum()} spam")
                
                return df_standardized
            else:
                print("❌ Could not identify text and label columns automatically")
                return create_sample_enron_data()
        else:
            print("📋 Dataset file not found, using sample data...")
            return create_sample_enron_data()
    
    except Exception as e:
        print(f"❌ Error loading Enron data: {e}")
        return create_sample_enron_data()

def create_sample_enron_data():
    """Create sample data that mimics Enron dataset patterns"""
    print("📋 Creating sample Enron-style data...")
    
    # Based on common Enron email patterns
    enron_spam_samples = [
        "win money now free energy trading bonus limited time offer",
        "claim your prize you won natural gas futures lottery",
        "urgent investment opportunity in oil and gas sector high returns",
        "exclusive energy stock tips buy now price will increase",
        "free electricity trading seminar sign up now limited seats",
        "work from home earn money energy market analysis",
        "million dollar energy investment opportunity act fast",
        "risk free power grid investment guaranteed returns",
        "special discount on energy derivatives trading platform",
        "congratulations selected for exclusive energy sector offer"
    ]
    
    enron_ham_samples = [
        "meeting scheduled for energy trading review tomorrow 10am conference room",
        "quarterly financial reports attached for review and approval",
        "power plant maintenance schedule needs updating please check",
        "natural gas pricing analysis completed and ready for presentation",
        "energy market trends discussion meeting rescheduled to Friday",
        "project deadline extension request has been approved by management",
        "client presentation materials are ready for download from server",
        "team building event next Friday all employees must attend",
        "budget approval required for q3 energy infrastructure projects",
        "contract negotiation update with our energy partners went well"
    ]
    
    # Combine data
    texts = enron_spam_samples + enron_ham_samples
    labels = [1] * len(enron_spam_samples) + [0] * len(enron_ham_samples)
    
    df = pd.DataFrame({'text': texts, 'label': labels})
    print(f"✅ Created sample dataset: {len(df)} emails")
    return df

def create_enron_spam_model():
    """Create spam detection model trained on Enron dataset patterns"""
    print("🔄 Creating Enron spam detection model...")
    
    # Load Enron data
    df = load_enron_data()
    
    # Prepare features and labels
    X = df['text'].values
    y = df['label'].values
    
    print(f"📊 Dataset balance: {sum(y)} spam, {len(y)-sum(y)} ham")
    print(f"📊 Spam percentage: {(sum(y)/len(y))*100:.1f}%")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📚 Training set: {len(X_train)} emails")
    print(f"🧪 Test set: {len(X_test)} emails")
    
    # Create pipeline with Enron-specific parameters
    model = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            sublinear_tf=True
        )),
        ('classifier', LogisticRegression(
            random_state=42,
            max_iter=1000,
            C=1.0,
            class_weight='balanced'
        ))
    ])
    
    # Train the model
    print("🎯 Training model on Enron data...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Model trained with {accuracy:.1%} accuracy")
    print("📈 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))
    
    # Test some predictions
    print("🧪 Testing model with sample texts:")
    test_texts = [
        "win free money now",
        "meeting tomorrow at 10am",
        "energy trading opportunity high returns"
    ]
    for text in test_texts:
        pred = model.predict([text])[0]
        proba = model.predict_proba([text])[0]
        print(f"   '{text}' -> {'SPAM' if pred == 1 else 'HAM'} (confidence: {max(proba):.2f})")
    
    # Save the model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    print("💾 Enron spam model saved successfully!")
    return model

# Load or create model
try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            spam_model = pickle.load(f)
        print("✅ Enron spam model loaded successfully!")
        
        # Test the loaded model
        test_text = "free energy trading opportunity"
        pred = spam_model.predict([test_text])[0]
        print(f"🧪 Model test: '{test_text}' -> {'SPAM' if pred == 1 else 'HAM'}")
        
    else:
        print("📝 No existing model found, creating new one...")
        spam_model = create_enron_spam_model()
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("🔄 Creating new model...")
    spam_model = create_enron_spam_model()

# --- Flask Setup ---
app = Flask(__name__)
app.secret_key = "enron_spam_detector_2025"

class EnronSpamPreprocessor:
    def __init__(self):
        # Enron-specific spam patterns
        self.enron_spam_patterns = [
            # Energy trading related spam
            'energy trading', 'natural gas', 'power grid', 'electricity market',
            'energy sector', 'futures trading', 'energy derivatives', 'power plant',
            'energy investment', 'oil and gas', 'renewable energy', 'energy stocks',
            'energy market', 'trading opportunity', 'energy portfolio',
            
            # Financial spam patterns
            'investment opportunity', 'high returns', 'stock tip', 'buy now',
            'price increase', 'insider information', 'secure returns',
            'financial gain', 'profit opportunity', 'money making',
            
            # General spam patterns
            'win', 'free', 'limited time', 'urgent', 'exclusive', 'special offer',
            'sign up', 'subscribe', 'click here', 'act now', 'don\'t miss',
            'last chance', 'risk free', 'guarantee', 'bonus', 'discount'
        ]

    def preprocess(self, text):
        """Preprocess text for Enron spam detection"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z\s\.!?]', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text

preprocessor = EnronSpamPreprocessor()

def predict_spam(text):
    """Predict spam using the Enron-trained model"""
    if spam_model is None:
        return "Error: Model not loaded", 0.0
    
    try:
        # Preprocess text
        cleaned_text = preprocessor.preprocess(text)
        
        if not cleaned_text.strip():
            return "not spam", 0.5
        
        # Make prediction
        prediction = spam_model.predict([cleaned_text])[0]
        confidence_scores = spam_model.predict_proba([cleaned_text])[0]
        confidence = max(confidence_scores)
        
        result = "spam" if prediction == 1 else "not spam"
        
        print(f"🔍 Enron Analysis: '{text[:50]}...' -> {result} (confidence: {confidence:.2f})")
        
        return result, confidence
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return "not spam", 0.5

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_single():
    """Predict spam for a single message input"""
    try:
        if not request.is_json:
            return jsonify({'error': 'Invalid request - JSON expected'}), 400
        
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        print(f"📨 Received text for Enron analysis: {text[:100]}...")
        
        prediction, confidence = predict_spam(text)
        
        return jsonify({
            'prediction': prediction,
            'confidence': f"{confidence:.1%}",
            'confidence_value': confidence
        })
        
    except Exception as e:
        print(f"❌ Route error: {e}")
        return jsonify({
            'prediction': 'not spam',
            'confidence': '50.0%',
            'confidence_value': 0.5
        })

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/stats')
def stats():
    return render_template('stats.html')

@app.route('/dataset_info')
def dataset_info():
    """Endpoint to show dataset information"""
    try:
        df = pd.read_csv(DATA_PATH)
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'sample_data': df.head(3).to_dict('records'),
            'column_types': str(df.dtypes.to_dict())
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/test_predictions')
def test_predictions():
    """Test endpoint with various examples"""
    test_cases = [
        "win free money now click here",
        "meeting tomorrow at 10am in conference room",
        "energy trading opportunity high returns investment",
        "please review the attached quarterly report",
        "urgent investment in natural gas futures"
    ]
    
    results = []
    for text in test_cases:
        prediction, confidence = predict_spam(text)
        results.append({
            'text': text,
            'prediction': prediction,
            'confidence': f"{confidence:.1%}"
        })
    
    return jsonify({'test_results': results})

if __name__ == '__main__':
    print("🚀 Starting Enron SpamGuard Server...")
    print("📍 Access the application at: http://localhost:5000")
    print("🔍 Dataset info: http://localhost:5000/dataset_info")
    print("🧪 Test predictions: http://localhost:5000/test_predictions")
    app.run(debug=True, port=5000, host='0.0.0.0')