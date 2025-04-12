import json
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load and preprocess data
def load_data(file_path):
    with open(file_path, 'r') as f:
        datastore = [json.loads(line) for line in f]
    
    df = pd.DataFrame(datastore)
    df = df[['is_sarcastic', 'headline']]
    
    # Clean text (same as previous preprocessing)
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.strip()
    
    df['clean_headline'] = df['headline'].apply(clean_text)
    return df

# Parameters
MAX_FEATURES = 10000  # Similar to VOCAB_SIZE in DL models
N_ESTIMATORS = 300    # Number of trees
N_JOBS = -1           # Use all cores

# Load data
df = load_data('Sarcasm_Headlines_Dataset_v2.json')

# TF-IDF Vectorization
tfidf = TfidfVectorizer(
    max_features=MAX_FEATURES,
    ngram_range=(1, 2),    # Include bigrams
    stop_words='english'
)

X = tfidf.fit_transform(df['clean_headline'])
y = df['is_sarcastic']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

# Random Forest Classifier
rf = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=15,
    min_samples_split=5,
    class_weight='balanced',
    n_jobs=N_JOBS,
    random_state=42
)

# Train
print("Training Random Forest...")
rf.fit(X_train, y_train)

# Evaluation
print("\nTest Set Performance:")
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature Importance
print("\nTop 10 Important Features:")
feature_names = tfidf.get_feature_names_out()
importances = rf.feature_importances_
indices = np.argsort(importances)[-10:][::-1]
for i in indices:
    print(f"{feature_names[i]}: {importances[i]:.4f}")

# Save model
joblib.dump(rf, 'sarcasm_rf.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')