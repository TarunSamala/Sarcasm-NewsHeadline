import json
import pandas as pd
import numpy as np
import re
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
DATA_PATH = '../Dataset/Sarcasm_Headlines_Dataset_v2.json'
OUTPUT_DIR = "sarcasm_outputs"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_text(text):
    """Basic text cleaning"""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_data(file_path):
    """Load and preprocess dataset"""
    with open(file_path, 'r') as f:
        datastore = [json.loads(line) for line in f]
    
    df = pd.DataFrame(datastore)
    df = df[['is_sarcastic', 'headline']]
    df['clean_headline'] = df['headline'].apply(clean_text)
    return df

# Load and split data
df = load_data(DATA_PATH)
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_headline'], 
    df['is_sarcastic'], 
    test_size=0.2, 
    random_state=42,
    stratify=df['is_sarcastic']
)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, 
                           class_weight='balanced',
                           n_jobs=-1,
                           random_state=42)
rf.fit(X_train_tfidf, y_train)

# Generate predictions
y_pred = rf.predict(X_test_tfidf)

# Save classification report
report = classification_report(y_test, y_pred)
with open(os.path.join(OUTPUT_DIR, 'classification_report-rf.txt'), 'w') as f:
    f.write("Random Forest Classification Report:\n")
    f.write(report)

# Save confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Sarcastic', 'Sarcastic'],
            yticklabels=['Non-Sarcastic', 'Sarcastic'])
plt.title('Random Forest Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix-rf.png'), bbox_inches='tight', dpi=300)
plt.close()

print("Results saved to:", OUTPUT_DIR)