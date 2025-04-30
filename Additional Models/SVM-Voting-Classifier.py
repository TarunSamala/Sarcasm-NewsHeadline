import json
import pandas as pd
import numpy as np
import re
import os
import joblib
import pkg_resources
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import loguniform
from matplotlib.backends.backend_pdf import PdfPages

# Configuration
OUTPUT_DIR = "sarcasm_outputs"
SAMPLE_SIZE = 10000
N_ITER = 15
CV_FOLDS = 2

# Create output directories
SUB_DIRS = [
    'metadata', 
    'performance/individual_metrics',
    'visualizations', 
    'models/individual_models',
    'reproducibility'
]

for path in SUB_DIRS:
    os.makedirs(os.path.join(OUTPUT_DIR, path), exist_ok=True)

# Updated JSON serializer
def json_serializer(obj):
    """Handle non-serializable objects for JSON dumping"""
    if isinstance(obj, (np.integer, np.int64, np.int32, int)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32, float)):
        return float(obj)
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    if isinstance(obj, type):
        return obj.__name__
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj  # Default handler for other types

# Text preprocessing
PATTERN = re.compile(r'[^a-zA-Z\s]')
def clean_text(text):
    text = str(text).lower()
    text = PATTERN.sub('', text)
    return ' '.join(text.split())

# Data loading
def load_data(file_path):
    with open(file_path, 'r') as f:
        datastore = [json.loads(line) for line in f]
    df = pd.DataFrame(datastore)
    df = df[['is_sarcastic', 'headline']].sample(SAMPLE_SIZE, random_state=42)
    df['clean_headline'] = df['headline'].apply(clean_text)
    return df

# Feature engineering pipeline
def create_feature_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            sublinear_tf=True
        ))
    ])

# Main execution
def main():
    # Load and split data
    df = load_data('../Dataset/Sarcasm_Headlines_Dataset_v2.json')
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_headline'], 
        df['is_sarcastic'], 
        test_size=0.2, 
        random_state=42
    )

    # Create features
    feature_pipeline = create_feature_pipeline()
    X_train_tfidf = feature_pipeline.fit_transform(X_train)
    X_test_tfidf = feature_pipeline.transform(X_test)

    # Initialize models
    models = {
        'svm': SVC(probability=True, class_weight='balanced', random_state=42),
        'lr': LogisticRegression(class_weight='balanced', solver='saga',
                               max_iter=1000, random_state=42, n_jobs=-1),
        'rf': RandomForestClassifier(class_weight='balanced', n_estimators=50,
                                   random_state=42, n_jobs=-1)
    }

    # Voting classifier
    voting_clf = VotingClassifier(
        estimators=list(models.items()), 
        voting='soft',
        n_jobs=-1
    )

    # Hyperparameter search
    param_dist = {
        'svm__C': loguniform(1e-3, 1e3),
        'svm__kernel': ['linear', 'rbf'],
        'lr__C': loguniform(1e-3, 1e3),
        'rf__max_depth': [None, 10, 20],
        'rf__min_samples_split': [2, 5, 10]
    }

    search = RandomizedSearchCV(
        voting_clf,
        param_dist,
        n_iter=N_ITER,
        cv=CV_FOLDS,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    # Train model
    print("Starting optimized training...")
    search.fit(X_train_tfidf, y_train)
    best_model = search.best_estimator_

    # Save full pipeline
    full_pipeline = Pipeline([
        ('features', feature_pipeline),
        ('model', best_model)
    ])
    joblib.dump(full_pipeline, os.path.join(OUTPUT_DIR, 'models/full_pipeline.joblib'))

    # Save individual models
    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        joblib.dump(model, os.path.join(OUTPUT_DIR, f'models/individual_models/{name}.joblib'))

    # Generate predictions
    y_pred = best_model.predict(X_test_tfidf)
    y_proba = best_model.predict_proba(X_test_tfidf)[:, 1]

    # Prepare metadata
    metadata = {
        "best_params": search.best_params_,
        "feature_config": feature_pipeline.named_steps['tfidf'].get_params(),
        "model_configs": {name: model.get_params() for name, model in models.items()},
        "environment": [str(pkg) for pkg in pkg_resources.working_set]
    }

    # Save metadata with custom serializer
    with open(os.path.join(OUTPUT_DIR, 'metadata/config.json'), 'w') as f:
        json.dump(metadata, f, indent=2, default=json_serializer)

    # Save performance metrics
    with open(os.path.join(OUTPUT_DIR, 'performance/full_report.txt'), 'w') as f:
        f.write(classification_report(y_test, y_pred))
        f.write(f"\nBest Parameters: {search.best_params_}")

    # Save individual model reports
    for name, model in models.items():
        report = classification_report(y_test, model.predict(X_test_tfidf))
        with open(os.path.join(OUTPUT_DIR, f'performance/individual_metrics/{name}_report.txt'), 'w') as f:
            f.write(f"{name.upper()} Report\n{report}")

    # Visualizations (same as before)
    # ... [rest of the visualization code]

    print(f"All results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()