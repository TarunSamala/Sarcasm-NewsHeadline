import os
import json
import re
import numpy as np
import pandas as pd
import tensorflow as tf
import pennylane as qml
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ===== CONFIG =====
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

VOCAB_SIZE = 15000
MAX_LENGTH = 40
EMBEDDING_DIM = 64
BATCH_SIZE = 32
EPOCHS = 10
QUBITS = 4
Q_LAYERS = 2
OUTPUT_DIR = "sarcasm_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== DATA CLEANING =====
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_data(file_path):
    with open(file_path, 'r') as f:
        datastore = [json.loads(line) for line in f]
    df = pd.DataFrame(datastore)
    df = df[['is_sarcastic', 'headline']]
    df['clean_headline'] = df['headline'].apply(clean_text)
    return df

# ===== QUANTUM CIRCUIT =====
dev = qml.device("default.qubit", wires=QUBITS)

@qml.qnode(dev, interface="tf")
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(QUBITS))
    qml.templates.BasicEntanglerLayers(weights, wires=range(QUBITS))
    return [qml.expval(qml.PauliZ(w)) for w in range(QUBITS)]

weight_shapes = {"weights": (Q_LAYERS, QUBITS)}
qlayer = qml.qnn.KerasLayer(quantum_circuit, weight_shapes, output_dim=QUBITS)

# ===== MODEL =====
def build_q_lstm_model():
    inputs = tf.keras.Input(shape=(MAX_LENGTH,))
    x = tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM)(inputs)
    x = tf.keras.layers.SpatialDropout1D(0.3)(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(32, return_sequences=False, dropout=0.3)
    )(x)

    # Reduce to match QUBITS for quantum layer
    x = tf.keras.layers.Dense(QUBITS)(x)

    x = qlayer(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
        metrics=["accuracy"]
    )
    return model

# ===== MAIN =====
if __name__ == "__main__":
    df = load_data("../Dataset/Sarcasm_Headlines_Dataset_v2.json")

    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_headline'],
        df['is_sarcastic'],
        test_size=0.2,
        random_state=42
    )

    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>', filters='')
    tokenizer.fit_on_texts(X_train)

    train_sequences = tokenizer.texts_to_sequences(X_train)
    test_sequences = tokenizer.texts_to_sequences(X_test)

    train_padded = pad_sequences(train_sequences, maxlen=MAX_LENGTH, padding='post')
    test_padded = pad_sequences(test_sequences, maxlen=MAX_LENGTH, padding='post')

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=3, min_delta=0.001, restore_best_weights=True
    )

    model = build_q_lstm_model()
    history = model.fit(
        train_padded, y_train,
        validation_data=(test_padded, y_test),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stop]
    )

    # ===== PLOTS =====
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

    plt.savefig("sarcasm_outputs/training_curves.png")
    plt.close()

    # ===== REPORTS =====
    y_pred = (model.predict(test_padded) > 0.5).astype(int).flatten()
    report = classification_report(y_test, y_pred)
    with open("sarcasm_outputs/classification_report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues',
                xticklabels=["Non-Sarcastic", "Sarcastic"],
                yticklabels=["Non-Sarcastic", "Sarcastic"],
                annot_kws={"size": 22})
    
    plt.title("Confusion Matrix",fontsize = 20)
    plt.xlabel("Predicted Label",fontsize = 20)
    plt.ylabel("True Label",fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ All results saved to 'sarcasm_outputs' folder.")
