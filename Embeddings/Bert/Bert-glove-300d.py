import json
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Create output directory
os.makedirs('sarcasm_outputs', exist_ok=True)

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

# Load and preprocess data
def load_data(file_path):
    with open(file_path, 'r') as f:
        datastore = [json.loads(line) for line in f]
    df = pd.DataFrame(datastore)
    df = df[['is_sarcastic', 'headline']]
    df['clean_headline'] = df['headline'].apply(clean_text)
    return df

# Parameters
MAX_LENGTH = 40
VOCAB_SIZE = 20000  # Max vocabulary size for TextVectorization
EMBED_DIM = 300     # GloVe embedding dimension
BATCH_SIZE = 8
EPOCHS = 4

if __name__ == "__main__":
    # Load and split data
    df = load_data('../../Dataset/Sarcasm_Headlines_Dataset_v2.json')
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_headline'],
        df['is_sarcastic'],
        test_size=0.2,
        random_state=42
    )

    # Compute class weights
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights_dict = dict(zip(classes, class_weights))
    sample_weights = np.array([class_weights_dict[label] for label in y_train])

    # Text Vectorization Layer
    vectorizer = TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_sequence_length=MAX_LENGTH,
        standardize=None,         # Already cleaned
        split='whitespace',       # Split on whitespace
        output_mode='int'
    )
    vectorizer.adapt(X_train)
    vocab = vectorizer.get_vocabulary()
    vocab_size = len(vocab)

    # Load GloVe embeddings
    print("Loading GloVe embeddings...")
    with open('../../../glove.840B.300d.pkl', 'rb') as f:
        embeddings_index = pickle.load(f)

    # Build embedding matrix
    print("Building embedding matrix...")
    embedding_matrix = np.zeros((vocab_size, EMBED_DIM))
    hits = 0
    misses = 0
    for i, word in enumerate(vocab):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print(f"Converted {hits}/{vocab_size} words (misses: {misses})")

    # Convert texts to sequences
    X_train_seq = vectorizer(X_train).numpy()
    X_test_seq = vectorizer(X_test).numpy()

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((
        X_train_seq, y_train, sample_weights
    )).shuffle(1000).batch(BATCH_SIZE)

    test_dataset = tf.data.Dataset.from_tensor_slices((
        X_test_seq, y_test
    )).batch(BATCH_SIZE)

    # Build model
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=EMBED_DIM,
            embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
            trainable=False,
            name='glove_embedding'
        ),
        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    # Compile model
    optimizer = AdamW(learning_rate=2e-5, weight_decay=0.05)
    loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Define callbacks
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        min_delta=0.001,
        restore_best_weights=True
    )
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )

    # Train model
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=EPOCHS,
        callbacks=[early_stop, lr_scheduler]
    )

    # Save training curves
    def save_training_curves(history):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Accuracy Curves')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Loss Curves')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(os.path.join('sarcasm_outputs', 'training_curves.png'), bbox_inches='tight')
        plt.close()

    save_training_curves(history)

    # Generate predictions
    probs = model.predict(test_dataset)
    y_pred = (probs > 0.5).astype(int)

    # Save classification report
    report = classification_report(y_test, y_pred)
    with open(os.path.join('sarcasm_outputs', 'classification_report.txt'), 'w') as f:
        f.write("Classification Report:\n")
        f.write(report)

    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Sarcastic', 'Sarcastic'],
                yticklabels=['Non-Sarcastic', 'Sarcastic'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join('sarcasm_outputs', 'confusion_matrix.png'), bbox_inches='tight')
    plt.close()

    print("All results saved to 'sarcasm_outputs' directory")