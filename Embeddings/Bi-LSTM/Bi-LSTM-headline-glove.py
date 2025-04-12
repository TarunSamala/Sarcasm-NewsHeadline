import re
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datasets import load_dataset
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Configuration
MAX_LEN = 50  # Reduced sequence length
VOCAB_SIZE = 25000  # Increased vocabulary size
EMBEDDING_DIM = 128
BATCH_SIZE = 256
EPOCHS = 10

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()

# Load and preprocess data
dataset = load_dataset("daniel2588/sarcasmdata")
train = dataset['train']
test = dataset['test']

# Clean texts
train_texts = [clean_text(t) for t in train['text']]
test_texts = [clean_text(t) for t in test['text']]
train_labels = np.array(train['label'])
test_labels = np.array(test['label'])

# Tokenization
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(train_texts)

# Sequencing
X_train = pad_sequences(tokenizer.texts_to_sequences(train_texts), maxlen=MAX_LEN, padding='post')
X_test = pad_sequences(tokenizer.texts_to_sequences(test_texts), maxlen=MAX_LEN, padding='post')

# Balanced Bi-LSTM Model
def build_balanced_model():
    inputs = Input(shape=(MAX_LEN,))
    
    # Embedding layer with tighter regularization
    x = Embedding(VOCAB_SIZE, EMBEDDING_DIM,
                  embeddings_regularizer=regularizers.l2(1e-4),
                  mask_zero=True)(inputs)
    
    x = SpatialDropout1D(0.3)(x)  # Reduced spatial dropout
    
    # Stacked Bi-LSTMs with progressive dimensionality reduction
    x = Bidirectional(LSTM(64, 
                          return_sequences=True,
                          kernel_regularizer=regularizers.l2(1e-4),
                          recurrent_regularizer=regularizers.l2(1e-4),
                          dropout=0.2,
                          recurrent_dropout=0.2))(x)
    
    x = Bidirectional(LSTM(32,
                          kernel_regularizer=regularizers.l2(1e-4),
                          recurrent_regularizer=regularizers.l2(1e-4),
                          dropout=0.2,
                          recurrent_dropout=0.2))(x)
    
    # Dense layer with moderate regularization
    x = Dense(64, activation='relu', 
             kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Dropout(0.4)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),  # Lower learning rate
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Enhanced callbacks
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5, 
                 min_delta=0.001, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                     patience=2, min_lr=1e-5)
]

# Training
model = build_balanced_model()
history = model.fit(
    X_train, train_labels,
    validation_data=(X_test, test_labels),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# Evaluation
test_loss, test_acc = model.evaluate(X_test, test_labels, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")