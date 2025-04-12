import json
import pandas as pd
import numpy as np
import tensorflow as tf
import re
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import SpatialDropout1D, LSTM, Bidirectional, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Constant
from tensorflow.keras.regularizers import l2

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
def build_optimized_model():
    inputs = Input(shape=(MAX_LEN,))
    
    # Embedding layer with regularization
    x = Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBEDDING_DIM,
        embeddings_initializer=Constant(embedding_matrix),
        trainable=False,
        mask_zero=True
    )(inputs)
    
    # Spatial dropout for sequence data
    x = SpatialDropout1D(0.3)(x)
    
    # First BiLSTM layer with regularization
    x = Bidirectional(LSTM(
        128,
        return_sequences=True,
        recurrent_dropout=0.2,
        kernel_regularizer=l2(0.001),
        merge_mode='concat'
    )(x)
    
    # Attention layer with regularization
    x = AttentionLayer()(x)
    x = Dropout(0.4)(x)
    
    # Second dense layer with regularization
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0003),  # Lower learning rate
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Enhanced callbacks
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=7,
        min_delta=0.002,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=3,
        min_lr=1e-6
    ),
    ModelCheckpoint(
        'best_model.h5',
        save_best_only=True,
        monitor='val_accuracy'
    )
]

# Training with full validation set
history = model.fit(
    X_train, train_labels,
    validation_data=(X_test, test_labels),  # Use explicit validation data
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# Enhanced plotting function
def plot_enhanced_training(history):
    plt.figure(figsize=(14, 6))
    
    # Accuracy plot with improved styling
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train', linewidth=2, color='#1f77b4')
    plt.plot(history.history['val_accuracy'], label='Validation', linewidth=2, color='#ff7f0e')
    plt.title('Training vs Validation Accuracy', fontsize=14)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylim(0.5, 1.0)
    plt.grid(linestyle='--', alpha=0.6)
    plt.legend()
    
    # Loss plot with improved styling
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train', linewidth=2, color='#1f77b4')
    plt.plot(history.history['val_loss'], label='Validation', linewidth=2, color='#ff7f0e')
    plt.title('Training vs Validation Loss', fontsize=14)
    plt.ylabel('Loss', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylim(0, max(history.history['val_loss']) + 0.1)
    plt.grid(linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300)
    plt.close()