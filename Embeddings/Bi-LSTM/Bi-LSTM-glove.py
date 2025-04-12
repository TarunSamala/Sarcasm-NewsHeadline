import os
import numpy as np
import tensorflow as tf
import re
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datasets import load_dataset
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Configuration
MAX_LEN = 35  # Optimized sequence length
VOCAB_SIZE = 15000  # Reduced vocabulary size
EMBEDDING_DIM = 128
BATCH_SIZE = 128  # Smaller batch size
EPOCHS = 30

# Enhanced text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load and preprocess data
dataset = load_dataset("daniel2588/sarcasmdata")
train = dataset['train']
test = dataset['test']

train_texts = [clean_text(t) for t in train['text']]
test_texts = [clean_text(t) for t in test['text']]
train_labels = np.array(train['label'])
test_labels = np.array(test['label'])

# Tokenization
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(train_texts)

# Sequencing with post padding
X_train = pad_sequences(tokenizer.texts_to_sequences(train_texts), 
                        maxlen=MAX_LEN, padding='post')
X_test = pad_sequences(tokenizer.texts_to_sequences(test_texts), 
                       maxlen=MAX_LEN, padding='post')

# Optimized Model Architecture
def build_robust_model():
    inputs = Input(shape=(MAX_LEN,))
    
    # Embedding layer with regularization
    x = Embedding(VOCAB_SIZE, EMBEDDING_DIM,
                  embeddings_regularizer=regularizers.l2(1e-4))(inputs)
    
    x = SpatialDropout1D(0.4)(x)
    
    # Stacked Bi-LSTMs with careful regularization
    x = Bidirectional(LSTM(64, 
                          return_sequences=True,
                          kernel_regularizer=regularizers.l2(1e-4),
                          recurrent_regularizer=regularizers.l2(1e-4),
                          dropout=0.3,
                          recurrent_dropout=0.0))(x)
    
    x = Bidirectional(LSTM(32,
                          kernel_regularizer=regularizers.l2(1e-4),
                          recurrent_regularizer=regularizers.l2(1e-4),
                          dropout=0.3,
                          recurrent_dropout=0.0))(x)
    
    # Dense layer with strong regularization
    x = Dense(64, activation='relu', 
             kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # Lower learning rate
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Enhanced Callbacks
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=6, 
                 min_delta=0.001, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                     patience=3, min_lr=1e-6)
]

# Training
model = build_robust_model()
history = model.fit(
    X_train, train_labels,
    validation_data=(X_test, test_labels),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# Evaluation and Visualization
def plot_training(history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy Curves')
    plt.ylabel('Accuracy')
    plt.ylim(0.5, 1.0)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss Curves')
    plt.ylabel('Loss')
    plt.ylim(0, 1)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

plot_training(history)

# Generate reports
y_pred = (model.predict(X_test) > 0.5).astype(int)
print("\nClassification Report:")
print(classification_report(test_labels, y_pred, target_names=['Not Sarcastic', 'Sarcastic']))

plt.figure(figsize=(8, 6))
cm = confusion_matrix(test_labels, y_pred)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Not Sarcastic', 'Sarcastic'], rotation=45)
plt.yticks(tick_marks, ['Not Sarcastic', 'Sarcastic'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()