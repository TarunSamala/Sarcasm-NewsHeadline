import json
import pandas as pd
import numpy as np
import tensorflow as tf
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Suppress TensorFlow info logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

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

    df['clean_headline'] = df['headline'].apply(clean_text)
    return df

# Parameters
VOCAB_SIZE = 15000
MAX_LENGTH = 40
EMBEDDING_DIM = 128
BATCH_SIZE = 512 # Increased batch size
EPOCHS = 3        # reduced epoch

# Load and prepare data
df = load_data('Sarcasm_Headlines_Dataset_v2.json')
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

# Bi-GRU Model Architecture
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32),
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM),
    tf.keras.layers.SpatialDropout1D(0.5),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64,kernel_regularizer=tf.keras.regularizers.l2(0.01))),
    tf.keras.layers.Dropout(0.5), # Increased Dropout
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Build model explicitly
model.build(input_shape=(None, MAX_LENGTH))

# Compile
model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.0005),  # Lower learning rate
    metrics=['accuracy']
)

# Display parameter count
model.summary()

# Training with callbacks
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,
    min_lr=1e-6
)

history = model.fit(
    train_padded,
    y_train,
    epochs=EPOCHS,
    validation_data=(test_padded, y_test),
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, reduce_lr]
)

# Evaluation
def plot_training(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.legend()

    plt.show()

plot_training(history)

# Generate predictions
y_pred = (model.predict(test_padded) > 0.5).astype(int)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
