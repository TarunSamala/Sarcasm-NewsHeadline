import json
import pickle
import pandas as pd
import numpy as np
import re
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
DATA_PATH = '../../Dataset/Sarcasm_Headlines_Dataset_v2.json'
FASTTEXT_PATH = '../../../crawl-300d-2M.vec'  # Updated path
OUTPUT_DIR = "sarcasm_outputs"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_text(text):
    """Basic text cleaning"""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_fasttext_embeddings(file_path):
    """Load FastText embeddings from .vec file"""
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        # Handle header line with vocab size and dimensions
        first_line = f.readline().strip()
        if len(first_line.split()) == 2:
            vocab_size, dim = map(int, first_line.split())
        else:
            # If first line is not header, reset file pointer
            f.seek(0)
        
        for line in f:
            parts = line.rstrip().split(' ')
            word = parts[0].lower()  # Convert to lowercase for consistency
            vector = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vector
    return embeddings

def text_to_vector(text, embeddings):
    """Convert text to average vector"""
    words = text.split()
    vectors = [embeddings.get(word, np.zeros(300)) for word in words]
    return np.mean(vectors, axis=0) if vectors else np.zeros(300)

# Load and preprocess data
def load_data(file_path):
    with open(file_path, 'r') as f:
        datastore = [json.loads(line) for line in f]
    
    df = pd.DataFrame(datastore)
    df = df[['is_sarcastic', 'headline']]
    df['clean_headline'] = df['headline'].apply(clean_text)
    return df

# Load dataset
df = load_data(DATA_PATH)

# Load FastText embeddings
print("Loading FastText crawl-300d-2M embeddings...")
fasttext_embeddings = load_fasttext_embeddings(FASTTEXT_PATH)

# Convert texts to vectors
print("Converting texts to vectors...")
X = np.array([text_to_vector(text, fasttext_embeddings) for text in df['clean_headline']])
y = df['is_sarcastic'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

# Train Random Forest
print("Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)

# Generate predictions
y_pred = rf.predict(X_test)

# Save classification report
report = classification_report(y_test, y_pred)
with open(os.path.join(OUTPUT_DIR, 'classification_report-rf-crawl-300d.txt'), 'w') as f:
    f.write("Random Forest with FastText crawl-300d-2M Classification Report:\n")
    f.write(report)

# Save confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Sarcastic', 'Sarcastic'],
            yticklabels=['Non-Sarcastic', 'Sarcastic'])
plt.title('Random Forest with FastText crawl-300d-2M Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix-crawl-300d.png'), bbox_inches='tight', dpi=300)
plt.close()

print("Results saved to:", OUTPUT_DIR)