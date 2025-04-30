import os
import json
import pandas as pd
import numpy as np
import re
import tensorflow as tf
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
MAX_LEN = 30
BATCH_SIZE = 8
EPOCHS = 4
OUTPUT_DIR = "sarcasm_outputs_roberta"
MODEL_NAME = "roberta-base"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load and preprocess data
def load_data(file_path, sample_size=8000):
    with open(file_path, 'r') as f:
        datastore = [json.loads(line) for line in f]
    
    df = pd.DataFrame(datastore)
    df = df[['is_sarcastic', 'headline']].sample(sample_size, random_state=42)
    df['clean_headline'] = df['headline'].apply(clean_text)
    return df

# Memory-optimized Model
def build_roberta_cnn():
    # Load pre-trained RoBERTa
    roberta = TFRobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1,
        output_hidden_states=True
    )
    
    # Freeze layers
    for layer in roberta.layers[:-8]:
        layer.trainable = False
    
    # Input layers
    input_ids = layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_ids')
    attention_mask = layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name='attention_mask')
    
    # Get RoBERTa outputs
    outputs = roberta([input_ids, attention_mask])
    hidden_states = outputs.hidden_states[-2]
    
    # CNN architecture
    x = layers.Conv1D(32, 3, activation='relu')(hidden_states)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=[input_ids, attention_mask], outputs=outputs)
    
    # Mixed precision
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=2e-5,
        global_clipnorm=1.0
    )
    model.compile(
        optimizer=optimizer,
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

    # Initialize tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    # Tokenization
    train_encodings = tokenizer(
        X_train.tolist(),
        max_length=MAX_LEN,
        truncation=True,
        padding='max_length',
        return_tensors='tf'
    )
    
    test_encodings = tokenizer(
        X_test.tolist(),
        max_length=MAX_LEN,
        truncation=True,
        padding='max_length',
        return_tensors='tf'
    )

    # Corrected dataset creation
    train_dataset = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask']
        },
        np.array(y_train)
    )).shuffle(500).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': test_encodings['input_ids'],
            'attention_mask': test_encodings['attention_mask']
        },
        np.array(y_test)
    )).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=1,
                     min_delta=0.005, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                         patience=1, min_lr=1e-6)
    ]

    # Training
    model = build_roberta_cnn()
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
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
    y_pred = (model.predict(test_dataset) > 0.5).astype(int).flatten()

    # Classification Report
    report = classification_report(y_test, y_pred)
    with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
        f.write(report)

    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    plt.close()

    print(f"Results saved to {OUTPUT_DIR}")