import json
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import re
import os

# Suppress warnings and configure environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress AVX warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Verify environment
print(f"TensorFlow version: {tf.__version__}")
print(f"Physical devices: {tf.config.list_physical_devices('GPU')}")

# Load and preprocess data
def load_data(file_path):
    with open(file_path, 'r') as f:
        datastore = [json.loads(line) for line in f]
    
    df = pd.DataFrame(datastore)
    df = df[['is_sarcastic', 'headline']]
    
    # Clean text
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.strip()
    
    df['clean_text'] = df['headline'].apply(clean_text)
    return df

# Parameters
MAX_LENGTH = 64
BATCH_SIZE = 8
LEARNING_RATE = 3e-5
EPOCHS = 4

# Load data
df = load_data('Sarcasm_Headlines_Dataset_v2.json')
X = df['clean_text'].values
y = df['is_sarcastic'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

# Initialize BERT components
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tokenization
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

# Compile and train
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

print("\nStarting training...")
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=EPOCHS,
    callbacks=[early_stopping]
)

# Evaluation
print("\nFinal Evaluation:")
results = model.evaluate(test_dataset)
print(f"Test Loss: {results[0]:.4f}")
print(f"Test Accuracy: {results[1]:.4f}")

# Save model
model.save_pretrained('bert_sarcasm')
tokenizer.save_pretrained('bert_sarcasm')
print("\nModel saved successfully!")