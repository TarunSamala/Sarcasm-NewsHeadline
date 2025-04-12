import os
import json
import numpy as np
import tensorflow as tf
import re
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Constant
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ---------------------- Configuration ---------------------- #
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

MAX_LEN = 100
VOCAB_SIZE = 20000
EMBEDDING_DIM = 300
BATCH_SIZE = 128
EPOCHS = 30
OUTPUT_DIR = "sarcasm_crawl300d_outputs"
CRAWL_PATH = "../../../crawl-300d-2M.vec"
DATA_PATH = "../../Dataset/Sarcasm_Headlines_Dataset_v2.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_data(file_path):
    with open(file_path, 'r') as f:
        datastore = [json.loads(line) for line in f]
    data = []
    for entry in datastore:
        text = clean_text(entry.get("headline", ""))
        label = entry.get("is_sarcastic", 0)
        data.append((text, label))
    return data

data = load_data(DATA_PATH)
texts, labels = zip(*data)
labels = np.array(labels)

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

def load_crawl_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
            except ValueError:
                continue
            if coefs.shape[0] != EMBEDDING_DIM:
                continue
            embeddings_index[word] = coefs
    return embeddings_index

print("Loading crawl-300d-2M embeddings...")
crawl_embeddings = load_crawl_embeddings(CRAWL_PATH)
embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
word_index = tokenizer.word_index
for word, i in word_index.items():
    if i < VOCAB_SIZE and word in crawl_embeddings:
        embedding_matrix[i] = crawl_embeddings[word]

def build_crawl300d_model():
    inputs = Input(shape=(MAX_LEN,))
    x = Embedding(VOCAB_SIZE, EMBEDDING_DIM,
                  embeddings_initializer=Constant(embedding_matrix),
                  trainable=False,
                  mask_zero=True)(inputs)
    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(LSTM(128,
                           kernel_regularizer=regularizers.l2(0.001),
                           recurrent_regularizer=regularizers.l2(0.001),
                           dropout=0.2))(x)
    x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    return model

checkpoint_path = os.path.join(OUTPUT_DIR, 'best_model.keras')
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-6),
    ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy')
]


model = build_crawl300d_model()
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

def save_training_curves(history):
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    plt.title('Accuracy Curves', fontsize=14)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylim(0.5, 1.0)
    plt.grid(linestyle='--', alpha=0.6)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation', linewidth=2)
    plt.title('Loss Curves', fontsize=14)
    plt.ylabel('Loss', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylim(0, 1.0)
    plt.grid(linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

save_training_curves(history)

y_pred = (model.predict(X_test) > 0.5).astype(int)
report = classification_report(y_test, y_pred, target_names=['Not Sarcastic', 'Sarcastic'])
with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
    f.write("Classification Report:\n")
    f.write(report)

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Sarcastic', 'Sarcastic'],
            yticklabels=['Not Sarcastic', 'Sarcastic'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")
print(f"All outputs saved to: {os.path.abspath(OUTPUT_DIR)}")
