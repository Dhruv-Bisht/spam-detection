# create_dummy_model.py - Run this once to generate a basic fallback model
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Simple dummy training data (spam vs ham examples)
training_texts = [
    # Ham
    "Hi, how are you today?",
    "Meeting at 3pm tomorrow.",
    "Thanks for your email.",
    "I love you.",
    # Spam
    "Congratulations! You've won $1000!",
    "Click here to claim your free gift now!",
    "Urgent: Your account will be suspended.",
    "Limited time offer: Buy now and save!"
]
labels = [0, 0, 0, 0, 1, 1, 1, 1]  # 0=ham, 1=spam

# Create a simple pipeline model
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=1000)),
    ('clf', MultinomialNB())
])

model.fit(training_texts, labels)

# Save the model
joblib.dump(model, 'spam_model.pkl')
print("✅ Dummy spam_model.pkl created successfully!")