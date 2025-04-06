import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, TFBertForSequenceClassification
import re

# Function to load and preprocess data
def load_data(file_path):
    """Load JSONL data and clean the text."""
    with open(file_path, 'r') as f:
        datastore = [json.loads(line) for line in f]
    df = pd.DataFrame(datastore)
    df = df[['is_sarcastic', 'headline']]
    # Clean text: lowercase and remove non-alphabetic characters
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.strip()
    df['clean_headline'] = df['headline'].apply(clean_text)
    return df['clean_headline'].tolist(), df['is_sarcastic'].tolist()

# Load data with fallback to sample dataset
try:
    texts, labels = load_data('../Sarcasm_Headlines_Dataset_v2.json')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Dataset not found. Using sample data.")
    texts = [
        "I love this movie",
        "This film is terrible",
        "What a fantastic day",
        "This is the worst experience ever",
        "Absolutely wonderful",
        "I hate it here"
    ]
    labels = [1, 0, 1, 0, 1, 0]
labels = np.array(labels)

# --- GloVe with LSTM ---

# Hyperparameters
VOCAB_SIZE = 15000
MAX_LENGTH = 40
EMBEDDING_DIM = 100

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=MAX_LENGTH)

# Function to load GloVe embeddings
def load_glove_embeddings(glove_file_path):
    """Load GloVe embeddings from file into a dictionary."""
    embeddings = {}
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# Load GloVe embeddings and create embedding matrix
glove_embeddings = load_glove_embeddings('../../glove.6B.100d.txt')
word_index = tokenizer.word_index
embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
for word, i in word_index.items():
    if i < VOCAB_SIZE:
        embedding_vector = glove_embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# Build and compile the GloVe + LSTM model
model_glove = Sequential([
    Embedding(VOCAB_SIZE, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_LENGTH, trainable=False),
    LSTM(64),
    Dense(1, activation='sigmoid')
])
model_glove.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Split data for GloVe + LSTM
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train the GloVe + LSTM model
print("Training GloVe + LSTM model...")
model_glove.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# --- BERT ---

# Split texts and labels for BERT
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Tokenize with BERT tokenizer
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer_bert(train_texts, truncation=True, padding=True, max_length=128, return_tensors='tf')
test_encodings = tokenizer_bert(test_texts, truncation=True, padding=True, max_length=128, return_tensors='tf')

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels
)).shuffle(100).batch(16)

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    test_labels
)).batch(16)

# Load and compile BERT model
model_bert = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model_bert.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

# Train the BERT model
print("Training BERT model...")
model_bert.fit(train_dataset, epochs=3, validation_data=test_dataset)

# --- Evaluation ---

# Evaluate GloVe + LSTM
loss_glove, acc_glove = model_glove.evaluate(X_test, y_test)
print(f"GloVe + LSTM Accuracy: {acc_glove:.4f}")

# Evaluate BERT
loss_bert, acc_bert = model_bert.evaluate(test_dataset)
print(f"BERT Accuracy: {acc_bert:.4f}")

# --- Ensemble Predictions ---

# Get predictions from GloVe + LSTM
pred_glove = model_glove.predict(X_test).flatten()

# Get predictions from BERT
pred_bert = model_bert.predict(test_dataset).logits
pred_bert_probs = tf.nn.softmax(pred_bert, axis=1)[:, 1].numpy()

# Ensemble by averaging probabilities
ensemble_pred = (pred_glove + pred_bert_probs) / 2
ensemble_labels = (ensemble_pred > 0.5).astype(int)
ensemble_acc = accuracy_score(y_test, ensemble_labels)
print(f"Ensemble Accuracy: {ensemble_acc:.4f}")