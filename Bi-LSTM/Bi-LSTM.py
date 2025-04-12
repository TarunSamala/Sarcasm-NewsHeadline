import json
import pandas as pd
import numpy as np
import tensorflow as tf
import re
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress TensorFlow info logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Create output directory
os.makedirs('sarcasm_outputs', exist_ok=True)

# Enhanced text cleaning with better normalization
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

# Load and preprocess data
def load_data(file_path):
    with open(file_path, 'r') as f:
        datastore = [json.loads(line) for line in f]
    
    df = pd.DataFrame(datastore)
    df = df[['is_sarcastic', 'headline']]
    df['clean_headline'] = df['headline'].apply(clean_text)
    return df

# Parameters with regularization enhancements
VOCAB_SIZE = 15000
MAX_LENGTH = 40
EMBEDDING_DIM = 64  # Increased embedding dimension
BATCH_SIZE = 128
EPOCHS = 15  # Reduced epochs with better early stopping

# Enhanced Model Architecture
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(MAX_LENGTH,)), 
        tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM),
        tf.keras.layers.SpatialDropout1D(0.6),  # Increased dropout
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            64, 
            return_sequences=False,
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            recurrent_dropout=0.4
        )),
        tf.keras.layers.Dense(64, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.7),  # Increased dropout
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Nadam(learning_rate=2e-4),  # Changed optimizer
        metrics=['accuracy']
    )
    return model

# Main execution
if __name__ == "__main__":
    # Load and prepare data
    df = load_data('../Dataset/Sarcasm_Headlines_Dataset_v2.json')
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_headline'], 
        df['is_sarcastic'], 
        test_size=0.2, 
        random_state=42
    )

    # Tokenization with frequency filtering
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>', filters='')
    tokenizer.fit_on_texts(X_train)

    train_sequences = tokenizer.texts_to_sequences(X_train)
    test_sequences = tokenizer.texts_to_sequences(X_test)

    train_padded = pad_sequences(train_sequences, maxlen=MAX_LENGTH, padding='post')
    test_padded = pad_sequences(test_sequences, maxlen=MAX_LENGTH, padding='post')

    # Callbacks with tighter early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        min_delta=0.001,
        restore_best_weights=True
    )

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2
    )

    # Training with class weights
    class_weights = {0: 1.2, 1: 0.8}  # Adjust based on your class distribution
    
    model = build_model()
    history = model.fit(
        train_padded,
        y_train,
        epochs=EPOCHS,
        validation_data=(test_padded, y_test),
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, lr_scheduler],
        class_weight=class_weights
    )

    # Save training curves
    def save_training_curves(history):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Accuracy Curves')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Loss Curves')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.savefig(os.path.join('sarcasm_outputs', 'training_curves.png'), bbox_inches='tight')
        plt.close()

    save_training_curves(history)

    # Generate and save reports
    y_pred = (model.predict(test_padded) > 0.5).astype(int).flatten()

    # Classification Report
    report = classification_report(y_test, y_pred)
    with open(os.path.join('sarcasm_outputs', 'classification_report.txt'), 'w') as f:
        f.write("Classification Report:\n")
        f.write(report)

    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Sarcastic', 'Sarcastic'],
                yticklabels=['Non-Sarcastic', 'Sarcastic'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join('sarcasm_outputs', 'confusion_matrix.png'), bbox_inches='tight')
    plt.close()

    print("All results saved to 'sarcasm_outputs' directory")