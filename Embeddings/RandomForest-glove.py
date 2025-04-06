import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from datetime import datetime
import os
import re

# Load and preprocess data
def load_data(file_path):
    with open(file_path, 'r') as f:
        datastore = [json.loads(line) for line in f]
    
    df = pd.DataFrame(datastore)
    df = df[['is_sarcastic', 'headline']]
    
    # Clean text (same as original)
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.strip()
    
    df['clean_headline'] = df['headline'].apply(clean_text)
    return df

# Load GloVe embeddings
def load_glove_embeddings(glove_file_path):
    embeddings = {}
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# Compute average GloVe vector for a headline
def get_average_vector(headline, embeddings, embedding_dim):
    words = headline.split()
    vectors = [embeddings.get(word, np.zeros(embedding_dim)) for word in words]
    if vectors and any(np.any(v) for v in vectors):  # Check if any vector is non-zero
        return np.mean(vectors, axis=0)
    return np.zeros(embedding_dim)  # Return zero vector if no valid words

# Parameters
EMBEDDING_DIM = 100  # Matches glove.6B.100d.txt
GLOVE_FILE_PATH = '../../glove.6B.100d.txt'  # Update if path differs

# Load data
df = load_data('../Sarcasm_Headlines_Dataset_v2.json')

# Check class distribution (optional)
print("Class Distribution:")
print(df['is_sarcastic'].value_counts())

# Load GloVe embeddings
if not os.path.exists(GLOVE_FILE_PATH):
    raise FileNotFoundError(f"GloVe file not found at {GLOVE_FILE_PATH}. Download from http://nlp.stanford.edu/data/glove.6B.zip")
glove_embeddings = load_glove_embeddings(GLOVE_FILE_PATH)
print(f"Loaded {len(glove_embeddings)} GloVe embeddings.")

# Create feature vectors
X = np.array([get_average_vector(headline, glove_embeddings, EMBEDDING_DIM) 
              for headline in df['clean_headline']])
y = df['is_sarcastic'].values

# Split data (stratified to preserve class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# Evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Plot and save ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

# Save plot
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"roc_curve_{timestamp}.png"
save_path = os.path.join(plot_dir, filename)
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"ROC curve saved to {save_path}")
plt.close()  # Close plot to free memory
