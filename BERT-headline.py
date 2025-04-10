import json
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification, BertConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from nlpaug.augmenter.word import ContextualWordEmbsAug

# Suppress warnings and configure environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Verify environment
print(f"TensorFlow version: {tf.__version__}")
print(f"Physical devices: {tf.config.list_physical_devices('GPU')}")

# --- Data Loading & Augmentation ---
def clean_text(text, augment=True):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Simple data augmentation
    if augment:
        words = text.split()
        if len(words) > 1:
            # Random word order reversal
            if np.random.rand() > 0.7:
                words = words[::-1]
            # Random adjacent swap
            if np.random.rand() > 0.7 and len(words) > 2:
                i = np.random.randint(0, len(words)-1)
                words[i], words[i+1] = words[i+1], words[i]
                
    return ' '.join(words).strip()

def load_data(file_path):
    with open(file_path, 'r') as f:
        datastore = [json.loads(line) for line in f]
    
    df = pd.DataFrame(datastore)
    df = df[['is_sarcastic', 'headline']]
    df['clean_text'] = df['headline'].apply(lambda x: clean_text(x, augment=True))
    return df

# --- Model Configuration ---
def create_model():
    config = BertConfig.from_pretrained('bert-base-uncased',
                                      hidden_dropout_prob=0.3,
                                      attention_probs_dropout_prob=0.3,
                                      num_labels=2)
    
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)
    
    # Freeze first 6 layers
    for layer in model.layers[0].encoder.layer[:6]:
        layer.trainable = False
    
    # Modified classifier with regularization
    model.classifier = tf.keras.Sequential([
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation='gelu', 
                            kernel_regularizer=tf.keras.regularizers.l1_l2(0.01, 0.01)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(2, activation=None)
    ])
    
    return model

# --- Training Setup ---
def main():
    # Parameters
    MAX_LENGTH = 64
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-5
    EPOCHS = 8
    
    # Load data
    df = load_data('Sarcasm_Headlines_Dataset_v2.json')
    X = df['clean_text'].values
    y = df['is_sarcastic'].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Tokenization
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def tokenize_texts(texts):
        return tokenizer(
            texts.tolist(),
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors='tf'
        )
    
    train_encodings = tokenize_texts(X_train)
    test_encodings = tokenize_texts(X_test)
    
    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(train_encodings),
        y_train
    )).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((
        dict(test_encodings),
        y_test
    )).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # Model & Optimizer
    model = create_model()
    
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        epsilon=1e-08
    )
    
    # Learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=LEARNING_RATE,
        first_decay_steps=len(train_dataset)*2,
        t_mul=1.5,
        m_mul=0.85
    )
    
    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        min_delta=0.002,
        restore_best_weights=True
    )
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_model',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    # Compile
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    # Train
    print("\nStarting training...")
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=EPOCHS,
        callbacks=[early_stopping, checkpoint, 
                  tf.keras.callbacks.LearningRateScheduler(lr_schedule)],
        class_weight={0: 1.2, 1: 0.8}
    )
    
    # --- Post-Training ---
    # Load best model
    model = tf.keras.models.load_model('best_model')
    
    # Visualization
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()
    
    # Evaluation
    y_pred = model.predict(test_dataset)
    y_pred_classes = tf.argmax(y_pred.logits, axis=1).numpy()
    
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Not Sarcastic', 'Sarcastic'],
               yticklabels=['Not Sarcastic', 'Sarcastic'])
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes,
                               target_names=['Not Sarcastic', 'Sarcastic']))
    
    # Save final model
    model.save_pretrained('final_bert_model')
    tokenizer.save_pretrained('final_bert_model')

if __name__ == "__main__":
    main()