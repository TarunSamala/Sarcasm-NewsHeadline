import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import re
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Embedding, Conv1D, 
                                   LSTM, Dense, Dropout, 
                                   SpatialDropout1D, Bidirectional, 
                                   GlobalMaxPooling1D, concatenate)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
MAX_LEN = 35
VOCAB_SIZE = 20000
EMBEDDING_DIM = 100
BATCH_SIZE = 128
EPOCHS = 30
OUTPUT_DIR = "sarcasm_outputs_cnnlstm"
GLOVE_PATH = "../../glove.6B.100d.txt"  # Update with your path

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load and preprocess data
def load_data(file_path):
    with open(file_path, 'r') as f:
        datastore = [json.loads(line) for line in f]
    
    df = pd.DataFrame(datastore)
    df = df[['is_sarcastic', 'headline']]
    df['clean_headline'] = df['headline'].apply(clean_text)
    return df

# Load GloVe embeddings
def load_glove_embeddings():
    embeddings_index = {}
    with open(GLOVE_PATH, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

# CNN-LSTM Model
def build_cnn_lstm_model(embedding_matrix):
    input_layer = Input(shape=(MAX_LEN,))
    
    # Embedding layer with GloVe weights
    x = Embedding(VOCAB_SIZE, EMBEDDING_DIM,
                  weights=[embedding_matrix],
                  input_length=MAX_LEN,
                  trainable=False)(input_layer)
    
    # Spatial dropout for CNN
    x = SpatialDropout1D(0.3)(x)
    
    # CNN branches with different kernel sizes
    conv1 = Conv1D(128, 3, activation='relu', padding='same')(x)
    conv1 = GlobalMaxPooling1D()(conv1)
    
    conv2 = Conv1D(128, 5, activation='relu', padding='same')(x)
    conv2 = GlobalMaxPooling1D()(conv2)
    
    # LSTM branch
    lstm = Bidirectional(LSTM(64, return_sequences=True))(x)
    lstm = Bidirectional(LSTM(64))(lstm)
    
    # Concatenate features
    merged = concatenate([conv1, conv2, lstm])
    
    # Classification head
    x = Dense(128, activation='relu',
              kernel_regularizer=regularizers.l2(0.01))(merged)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=input_layer, outputs=output)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Main execution
if __name__ == "__main__":
    # Load data
    df = load_data('../Dataset/Sarcasm_Headlines_Dataset_v2.json')
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_headline'], 
        df['is_sarcastic'], 
        test_size=0.2, 
        random_state=42
    )

    # Tokenization
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)
    
    # Convert to sequences
    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN)
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=MAX_LEN)

    # Load GloVe embeddings
    embeddings_index = load_glove_embeddings()
    
    # Prepare embedding matrix
    embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
    for word, i in tokenizer.word_index.items():
        if i >= VOCAB_SIZE:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # Build model
    model = build_cnn_lstm_model(embedding_matrix)
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=5,
                     min_delta=0.001, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                         patience=3, min_lr=1e-6)
    ]

    # Class weights for imbalance
    class_weights = {0: 1.2, 1: 0.8}  # Adjust based on your data distribution

    # Training
    history = model.fit(
        X_train_seq, y_train,
        validation_data=(X_test_seq, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # Save training curves
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy Curves')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss Curves')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'))
    plt.close()

    # Generate predictions
    y_pred = (model.predict(X_test_seq) > 0.5).astype(int)

    # Classification Report
    report = classification_report(y_test, y_pred)
    with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
        f.write(report)

    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    plt.close()

    print(f"All results saved to {OUTPUT_DIR}")