import json
import pandas as pd
import numpy as np
import tensorflow as tf
import re
import os
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Embedding, Bidirectional, GRU, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Constant
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight

# Configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
MAX_LEN = 50
VOCAB_SIZE = 15000
EMBEDDING_DIM = 100
BATCH_SIZE = 128
EPOCHS = 30
GLOVE_PATH = '../../../glove.6B.100d.txt'
DATA_PATH = '../../Dataset/Sarcasm_Headlines_Dataset_v2.json'
OUTPUT_DIR = "sarcasm_outputs"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_text(text):
    """Enhanced text cleaning with special character handling"""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_data(file_path):
    """Load and preprocess dataset"""
    with open(file_path, 'r') as f:
        datastore = [json.loads(line) for line in f]
    
    df = pd.DataFrame(datastore)
    df = df[['is_sarcastic', 'headline']]
    df['clean_headline'] = df['headline'].apply(clean_text)
    return df

# Load and split data
df = load_data(DATA_PATH)
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_headline'], 
    df['is_sarcastic'], 
    test_size=0.2, 
    random_state=42,
    stratify=df['is_sarcastic']
)

# Tokenization
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>', filters='')
tokenizer.fit_on_texts(X_train)

# Sequence padding
X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN, padding='post')
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=MAX_LEN, padding='post')

# Load GloVe embeddings
def load_glove(file_path):
    embeddings = {}
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

print("Loading GloVe embeddings...")
glove_embeddings = load_glove(GLOVE_PATH)

# Create embedding matrix
embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
for word, i in tokenizer.word_index.items():
    if i < VOCAB_SIZE and word in glove_embeddings:
        embedding_matrix[i] = glove_embeddings[word]

# Optimized Bi-GRU Model
def build_bigru_model():
    inputs = Input(shape=(MAX_LEN,))
    
    x = Embedding(VOCAB_SIZE, EMBEDDING_DIM,
                embeddings_initializer=Constant(embedding_matrix),
                mask_zero=True,
                trainable=False)(inputs)
    
    x = SpatialDropout1D(0.4)(x)
    
    x = Bidirectional(GRU(96,
                        kernel_regularizer=regularizers.l2(1e-4),
                        recurrent_regularizer=regularizers.l2(1e-4),
                        return_sequences=True))(x)
    
    x = Bidirectional(GRU(64,
                        kernel_regularizer=regularizers.l2(1e-4)))(x)
    
    x = Dense(128, activation='relu',
             kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Dropout(0.6)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=2e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5,
                 min_delta=0.001, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                     patience=3, min_lr=1e-6)
]

# Class weights
class_weights = compute_class_weight('balanced',
                                    classes=np.unique(y_train),
                                    y=y_train)
class_weights = {i:w for i,w in enumerate(class_weights)}

# Training
model = build_bigru_model()
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
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Bi-GRU Accuracy Curves (GloVe 100D)')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Bi-GRU Loss Curves (GloVe 100D)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves-bigru-glove-100d.png'), bbox_inches='tight')
plt.close()

# Generate predictions
y_pred = (model.predict(X_test_seq) > 0.5).astype(int).flatten()

# Save classification report
report = classification_report(y_test, y_pred)
with open(os.path.join(OUTPUT_DIR, 'classification_report-bigru-glove-100d.txt'), 'w') as f:
    f.write("Bi-GRU with GloVe 100D Classification Report:\n")
    f.write(report)

# Save confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Sarcastic', 'Sarcastic'],
            yticklabels=['Non-Sarcastic', 'Sarcastic'])
plt.title('Bi-GRU Confusion Matrix (GloVe 100D)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix-bigru-glove-100d.png'), bbox_inches='tight')
plt.close()

print("\nAll results saved to:", os.path.abspath(OUTPUT_DIR))