import json
import pandas as pd
import numpy as np
import tensorflow as tf
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification, DistilBertConfig

# Suppress TensorFlow info logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Create output directory
os.makedirs('sarcasm_outputs', exist_ok=True)

# Text cleaning function
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

# Parameters
MAX_LENGTH = 40
BATCH_SIZE = 16
EPOCHS = 5

# Main execution
if __name__ == "__main__":
    # Load and split data
    df = load_data('../../Dataset/Sarcasm_Headlines_Dataset_v2.json')
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_headline'],
        df['is_sarcastic'],
        test_size=0.2,
        random_state=42
    )

    # Compute class weights based on training data
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights_dict = dict(zip(classes, class_weights))
    sample_weights = np.array([class_weights_dict[label] for label in y_train])

    # Tokenize data using DistilBERT tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(
        X_train.tolist(),
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
        return_tensors='tf'
    )
    test_encodings = tokenizer(
        X_test.tolist(),
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
        return_tensors='tf'
    )

    # Prepare TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask']
        },
        y_train,
        sample_weights
    )).shuffle(1000).batch(BATCH_SIZE)

    test_dataset = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': test_encodings['input_ids'],
            'attention_mask': test_encodings['attention_mask']
        },
        y_test
    )).batch(BATCH_SIZE)

    # Build and compile DistilBERT model with AdamW optimizer
    config = DistilBertConfig.from_pretrained('distilbert-base-uncased', num_labels=1, dropout=0.2)
    model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', config=config)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=2e-5, weight_decay=0.05)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Define callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        min_delta=0.001,
        restore_best_weights=True
    )
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )

    # Train the model
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=test_dataset,
        callbacks=[early_stop, lr_scheduler]
    )

    # Function to save training curves
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

    # Generate predictions
    logits = model.predict(test_dataset).logits
    probabilities = tf.sigmoid(logits).numpy().flatten()
    y_pred = (probabilities > 0.5).astype(int)

    # Save classification report
    report = classification_report(y_test, y_pred)
    with open(os.path.join('sarcasm_outputs', 'classification_report.txt'), 'w') as f:
        f.write("Classification Report:\n")
        f.write(report)

    # Save confusion matrix
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