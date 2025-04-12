import os
import re
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Embedding, Bidirectional, GRU, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ---------------------- Configuration ---------------------- #
MAX_LEN = 35               # Maximum sequence length
VOCAB_SIZE = 12000         # Maximum vocabulary size
EMBEDDING_DIM = 96         # Embedding dimensions (randomly initialized)
BATCH_SIZE = 128
EPOCHS = 40
OUTPUT_DIR = "bigru_arcasm_outputs"
DATA_PATH = "../Dataset/Sarcasm_Headlines_Dataset_v2.json"

# Create output directory if not exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------- Enhanced Text Cleaning ---------------------- #
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------------- Data Loading & Preprocessing ---------------------- #
def load_data(file_path):
    with open(file_path, 'r') as f:
        datastore = [json.loads(line) for line in f]
    data = []
    for entry in datastore:
        # Assume keys "headline" for text and "is_sarcastic" for label
        t = clean_text(entry.get("headline", ""))
        label = entry.get("is_sarcastic", 0)
        data.append((t, label))
    return data

data = load_data(DATA_PATH)
texts, labels = zip(*data)
labels = np.array(labels)

# Tokenization and sequence padding
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)
X = pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=MAX_LEN, padding='post')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# ---------------------- Bi-GRU Model Definition ---------------------- #
def build_bigru_model():
    inputs = Input(shape=(MAX_LEN,))
    
    # Embedding layer with L2 regularization on the weights
    x = Embedding(VOCAB_SIZE, EMBEDDING_DIM, embeddings_regularizer=regularizers.l2(1e-4))(inputs)
    
    # SpatialDropout for better generalization
    x = SpatialDropout1D(0.5)(x)
    
    # cuDNN-optimized Bi-GRU layer: using 48 units with L2 regularization and dropout to reduce overfitting
    x = Bidirectional(GRU(48,
                          activation='tanh',
                          recurrent_activation='sigmoid',
                          reset_after=True,
                          kernel_regularizer=regularizers.l2(1e-4),
                          recurrent_regularizer=regularizers.l2(1e-4),
                          dropout=0.4,
                          recurrent_dropout=0.0))(x)
    
    # A Dense layer with regularization and further dropout
    x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Dropout(0.6)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Nadam(learning_rate=2e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ---------------------- Callbacks ---------------------- #
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=7, min_delta=0.001, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
]

# ---------------------- Training ---------------------- #
model = build_bigru_model()
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# ---------------------- Save Training Curves ---------------------- #
plt.figure(figsize=(15, 6))

# Accuracy Curve
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy Curves', pad=10)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.ylim(0.5, 1.0)
plt.grid(linestyle='--', alpha=0.6)
plt.legend()

# Loss Curve
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss Curves', pad=10)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.ylim(0, 1.5)
plt.grid(linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=300, bbox_inches='tight')
plt.close()

# ---------------------- Evaluation & Report Generation ---------------------- #
# Generate predictions
y_pred = (model.predict(X_test) > 0.5).astype(int)

# Save classification report
report = classification_report(y_test, y_pred, target_names=['Not Sarcastic', 'Sarcastic'])
report_path = os.path.join(OUTPUT_DIR, 'classification_report.txt')
with open(report_path, 'w') as f:
    f.write("Classification Report:\n")
    f.write(report)

# Save confusion matrix
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

print(f"\nAll outputs saved to: {os.path.abspath(OUTPUT_DIR)}")
