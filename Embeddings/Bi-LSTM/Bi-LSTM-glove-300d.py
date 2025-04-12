import json
import pickle
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

# Configuration
VOCAB_SIZE = 20000
MAX_LENGTH = 50
EMBEDDING_DIM = 300
BATCH_SIZE = 64
EPOCHS = 20
GLOVE_PATH = '../../../glove.840B.300d.pkl'
DATA_PATH = '../../Dataset/Sarcasm_Headlines_Dataset_v2.json'
OUTPUT_DIR = "sarcasm_outputs"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_text(text):
    """Enhanced text cleaning with special character handling"""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabet characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
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

# Sequence conversion
train_sequences = tokenizer.texts_to_sequences(X_train)
test_sequences = tokenizer.texts_to_sequences(X_test)

train_padded = pad_sequences(train_sequences, maxlen=MAX_LENGTH, padding='post')
test_padded = pad_sequences(test_sequences, maxlen=MAX_LENGTH, padding='post')

# Load GloVe embeddings
print("Loading GloVe embeddings...")
with open(GLOVE_PATH, 'rb') as f:
    glove_embeddings = pickle.load(f)

# Create embedding matrix
embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
for word, i in tokenizer.word_index.items():
    if i < VOCAB_SIZE and word in glove_embeddings:
        embedding_matrix[i] = glove_embeddings[word]

def build_glove_model():
    """Enhanced Bi-LSTM model with regularization"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32),
        tf.keras.layers.Embedding(
            VOCAB_SIZE, EMBEDDING_DIM,
            embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
            trainable=True
        ),
        tf.keras.layers.SpatialDropout1D(0.6),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            96, 
            return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
            recurrent_dropout=0.4
        )),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            48,
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)
        )),
        tf.keras.layers.Dense(96, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.7),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.AdamW(learning_rate=3e-5),
        metrics=['accuracy']
    )
    return model

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        min_delta=0.002,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3
    )
]

# Class weights for imbalance handling
class_weights = {0: 1.2, 1: 0.8}  # Adjust based on your dataset distribution

# Training
model = build_glove_model()
history = model.fit(
    train_padded,
    y_train,
    epochs=EPOCHS,
    validation_data=(test_padded, y_test),
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
plt.title('Accuracy Curves (GloVe)')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss Curves (GloVe)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves-glove-300d.png'), bbox_inches='tight')
plt.close()

# Generate predictions
y_pred = (model.predict(test_padded) > 0.5).astype(int).flatten()

# Save classification report
report = classification_report(y_test, y_pred)
with open(os.path.join(OUTPUT_DIR, 'classification_report-glove-300d.txt'), 'w') as f:
    f.write("GloVe 840B Classification Report:\n")
    f.write(report)

# Save confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Sarcastic', 'Sarcastic'],
            yticklabels=['Non-Sarcastic', 'Sarcastic'])
plt.title('Confusion Matrix (GloVe 840B)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix-glove-300d.png'), bbox_inches='tight')
plt.close()

print("\nAll results saved to:", os.path.abspath(OUTPUT_DIR))