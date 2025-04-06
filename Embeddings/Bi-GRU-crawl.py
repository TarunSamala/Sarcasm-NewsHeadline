import json
import pandas as pd
import numpy as np
import tensorflow as tf
import re
import os
from datetime import datetime
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Load and preprocess data
def load_data(file_path):
    with open(file_path, 'r') as f:
        datastore = [json.loads(line) for line in f]
    
    df = pd.DataFrame(datastore)
    df = df[['is_sarcastic', 'headline']]
    
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.strip()
    
    df['clean_headline'] = df['headline'].apply(clean_text)
    return df

# Load FastText embeddings
def load_fasttext(filepath):
    embeddings_index = {}
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

# Parameters
MAX_LENGTH = 30
FASTTEXT_DIM = 300  # Updated for FastText dimension
BATCH_SIZE = 256
EPOCHS = 20
GLOVE_DIM = 100

# Load and prepare data
df = load_data('../Sarcasm_Headlines_Dataset_v2.json')
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_headline'], 
    df['is_sarcastic'], 
    test_size=0.2, 
    random_state=42
)

# Tokenization
tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)
VOCAB_SIZE = len(tokenizer.word_index) + 1

# Load FastText embeddings
print("Loading FastText embeddings...")
fasttext_embeddings = load_fasttext('../../crawl-300d-2M.vec')
print(f"Loaded {len(fasttext_embeddings)} word vectors")

# Create embedding matrix
embedding_matrix = np.zeros((VOCAB_SIZE, FASTTEXT_DIM))
found = 0
for word, i in tokenizer.word_index.items():
    if i < VOCAB_SIZE:
        embedding_vector = fasttext_embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            found += 1
print(f"Embedding coverage: {found/VOCAB_SIZE:.2%}")

# Sequence padding
train_sequences = tokenizer.texts_to_sequences(X_train)
test_sequences = tokenizer.texts_to_sequences(X_test)
train_padded = pad_sequences(train_sequences, maxlen=MAX_LENGTH, padding='post')
test_padded = pad_sequences(test_sequences, maxlen=MAX_LENGTH, padding='post')

# Model architecture updated for FastText
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=GLOVE_DIM,
        weights=[embedding_matrix],
        input_length=MAX_LENGTH,
        trainable=False
    ),
    tf.keras.layers.SpatialDropout1D(0.3),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
        96,  # Increased units for GRU
        return_sequences=True,
        recurrent_dropout=0.25,
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    )),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
        64,
        recurrent_dropout=0.25,
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    )),
    tf.keras.layers.Dense(96, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.35),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.0003),  # Reduced learning rate
    metrics=['accuracy']
)

# Enhanced callbacks
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=7,  # Increased patience
    min_delta=0.002,
    restore_best_weights=True
)

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,  # More conservative reduction
    patience=3,
    min_lr=1e-6
)

# Training
history = model.fit(
    train_padded,
    y_train,
    epochs=EPOCHS,
    validation_data=(test_padded, y_test),
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, lr_scheduler]
)

# Enhanced visualization
def plot_training(history):
    plt.figure(figsize=(14, 6))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    plt.title('Training vs Validation Accuracy', fontsize=14)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylim(0.5, 1.0)
    plt.grid(linestyle='--', alpha=0.6)
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation', linewidth=2)
    plt.title('Training vs Validation Loss', fontsize=14)
    plt.ylabel('Loss', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylim(0, max(history.history['val_loss']) + 0.1)
    plt.grid(linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    
    # Save plot with improved naming
    plot_dir = "training_plots"
    os.makedirs(plot_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(
        f"{plot_dir}/fasttext_training_GRU_{timestamp}.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

plot_training(history)

# Enhanced evaluation
y_pred = (model.predict(test_padded) > 0.5).astype(int)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))