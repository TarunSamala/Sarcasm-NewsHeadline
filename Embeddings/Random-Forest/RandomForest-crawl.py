import json
import pandas as pd
import numpy as np
import re
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load and preprocess data
def load_data(file_path):
    with open(file_path, 'r') as f:
        datastore = [json.loads(line) for line in f]
    
    df = pd.DataFrame(datastore)
    df = df[['is_sarcastic', 'headline']]
    
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.strip()
    
    df['clean_headline'] = df['headline'].apply(clean_text)
    return df

# Load FastText embeddings
def load_fasttext(filepath):
    embeddings_index = {}
    with open(filepath, encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading FastText"):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

# Convert text to averaged vectors
def text_to_vector(text, embeddings, dim=300):
    words = text.split()
    vectors = [embeddings[word] for word in words if word in embeddings]
    if len(vectors) == 0:
        return np.zeros(dim)
    return np.mean(vectors, axis=0)

# Parameters
RANDOM_STATE = 42
N_ESTIMATORS = 300
MAX_DEPTH = 50

# Load and prepare data
df = load_data('../Sarcasm_Headlines_Dataset_v2.json')

# Load FastText embeddings
print("\nLoading embeddings...")
ft_embeddings = load_fasttext('../../crawl-300d-2M.vec')

# Convert text to vectors
print("\nConverting text to vectors...")
X = np.array([text_to_vector(text, ft_embeddings) for text in tqdm(df['clean_headline'])])
y = df['is_sarcastic'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# Initialize and train Random Forest
print("\nTraining Random Forest...")
rf = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# Generate predictions
y_pred = rf.predict(X_test)

# Enhanced visualization
def plot_results(y_true, y_pred):
    plt.figure(figsize=(14, 6))
    
    # Confusion Matrix
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Not Sarcastic', 'Sarcastic'])
    plt.yticks(tick_marks, ['Not Sarcastic', 'Sarcastic'])
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    
    # Feature Importance (Top 20)
    plt.subplot(1, 2, 2)
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[-20:]
    plt.barh(range(20), importances[sorted_idx], align='center')
    plt.yticks(range(20), [f"Dim {i}" for i in sorted_idx])
    plt.title('Top 20 Important Embedding Dimensions', fontsize=14)
    plt.xlabel('Importance', fontsize=12)
    
    plt.tight_layout()
    
    # Save plot
    plot_dir = "rf_plots"
    os.makedirs(plot_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(
        f"{plot_dir}/rf_results_{timestamp}.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

plot_results(y_test, y_pred)

# Evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))