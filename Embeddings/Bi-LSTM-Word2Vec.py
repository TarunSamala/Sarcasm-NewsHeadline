import json
import pandas as pd
import numpy as np
import tensorflow as tf
import re
import matplotlib
matplotlib.use('Agg')  # Keep for WSL2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
import os
import gensim

# Suppress TensorFlow info logs
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

# Load Word2Vec embeddings
def load_word2vec(model_path):
    print("Loading Word2Vec model...")
    model = gensim.models.KeyedVectors.load_word2vec_format(
        model_path,
        binary=True,
        limit=500000  # Limit to 500,000 most frequent words
    )
    print(f"Loaded {len(model)} word vectors")
    return model

# Parameters
VOCAB_SIZE = 15000
MAX_LENGTH = 40
EMBEDDING_DIM = 300  # Matches Word2Vec dimension
BATCH_SIZE = 128
EPOCHS = 10
GENSIM_MODEL_PATH = '../../GoogleNews-vectors-negative300.bin'

# Load and prepare data
df = load_data('../Sarcasm_Headlines_Dataset_v2.json')
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_headline'], 
    df['is_sarcastic'], 
    test_size=0.2, 
    random_state=42
)

# Tokenization
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

train_sequences = tokenizer.texts_to_sequences(X_train)
test_sequences = tokenizer.texts_to_sequences(X_test)

train_padded = pad_sequences(train_sequences, maxlen=MAX_LENGTH, padding='post')
test_padded = pad_sequences(test_sequences, maxlen=MAX_LENGTH, padding='post')

# Load Word2Vec embeddings
word2vec_model = load_word2vec(GENSIM_MODEL_PATH)

# Prepare embedding matrix
embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
for word, i in tokenizer.word_index.items():
    if i < VOCAB_SIZE:
        try:
            embedding_vector = word2vec_model[word]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            # Handle out-of-vocabulary words
            embedding_matrix[i] = np.random.normal(scale=0.6, size=(EMBEDDING_DIM,))

# Model architecture with Word2Vec
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32),
    tf.keras.layers.Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBEDDING_DIM,
        weights=[embedding_matrix],
        trainable=False  # Set True to fine-tune embeddings
    ),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.0005),  # Lower learning rate
    metrics=['accuracy']
)

model.summary()

# Training with callbacks
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    train_padded,
    y_train,
    epochs=EPOCHS,
    validation_data=(test_padded, y_test),
    batch_size=BATCH_SIZE,
    callbacks=[early_stop]
)

# Save training graphs
def save_training_curves(history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plot_dir = os.path.expanduser('~/word2vec_plots')
    os.makedirs(plot_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"word2vec_training_{timestamp}.png"
    save_path = os.path.join(plot_dir, filename)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"\nSaved training curves to: {save_path}")

save_training_curves(history)

# Evaluation
y_pred = (model.predict(test_padded) > 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))