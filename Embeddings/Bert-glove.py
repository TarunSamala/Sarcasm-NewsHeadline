import json
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, Conv1D, GlobalMaxPooling1D

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- GloVe Implementation ---
def load_glove_embeddings(glove_path):
    embeddings_index = {}
    with open(glove_path, encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def create_glove_model(vocab_size, embedding_dim, max_length, embeddings_index):
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_length,
            weights=[embeddings_matrix],
            trainable=False
        ),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.5),
        Conv1D(64, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    return model

# --- Data Processing ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()

def load_data(file_path):
    with open(file_path, 'r') as f:
        datastore = [json.loads(line) for line in f]
    
    df = pd.DataFrame(datastore)
    df = df[['is_sarcastic', 'headline']]
    df['clean_text'] = df['headline'].apply(clean_text)
    return df

# --- Main Function ---
def main():
    # Parameters
    MAX_LENGTH = 64
    BATCH_SIZE = 256
    EMBEDDING_DIM = 100
    EPOCHS = 15
    
    # Load data
    df = load_data('Sarcasm_Headlines_Dataset_v2.json')
    X = df['clean_text'].values
    y = df['is_sarcastic'].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Text Vectorization
    vectorizer = TextVectorization(
        max_tokens=20000,
        output_sequence_length=MAX_LENGTH,
        standardize='lower_and_strip_punctuation'
    )
    vectorizer.adapt(X_train)
    
    # Load GloVe embeddings
    embeddings_index = load_glove_embeddings('glove.6B.100d.txt')
    
    # Create embedding matrix
    vocab = vectorizer.get_vocabulary()
    embeddings_matrix = np.zeros((len(vocab), EMBEDDING_DIM))
    for i, word in enumerate(vocab):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embeddings_matrix[i] = embedding_vector
    
    # Create model
    model = create_glove_model(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        max_length=MAX_LENGTH,
        embeddings_matrix=embeddings_matrix
    )
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Convert text to sequences
    X_train_vec = vectorizer(X_train).numpy()
    X_test_vec = vectorizer(X_test).numpy()
    
    # Train
    history = model.fit(
        X_train_vec, y_train,
        validation_data=(X_test_vec, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
        ]
    )
    
    # Visualization and evaluation (same as BERT version)
    # ... [Use same visualization code as previous example]

if __name__ == "__main__":
    main()