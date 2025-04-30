import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import re
import random
from deap import base, creator, tools, algorithms
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Embedding, Conv1D, LSTM, 
                                   Dense, Dropout, SpatialDropout1D,
                                   GlobalMaxPooling1D, Bidirectional)
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add this import at the top with other sklearn imports
from sklearn.metrics import classification_report, confusion_matrix

# Configuration
MAX_LEN = 35
VOCAB_SIZE = 20000
EMBEDDING_DIM = 100
BATCH_SIZE = 32
EPOCHS = 30
GLOVE_PATH = "../../glove.6B.100d.txt"
OUTPUT_DIR = "sarcasm_outputs_genetic_lstm_cnn"

# Genetic Algorithm Parameters
POPULATION_SIZE = 5
GENERATIONS = 2
CXPB = 0.5
MUTPB = 0.2

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Custom Text Cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# Data Loading
def load_data(file_path):
    with open(file_path, 'r') as f:
        datastore = [json.loads(line) for line in f]
    df = pd.DataFrame(datastore)
    df = df[['is_sarcastic', 'headline']].sample(10000, random_state=42)
    df['clean_headline'] = df['headline'].apply(clean_text)
    return df

# Genetic Optimization Setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def create_individual():
    return [
        random.choice([64, 128]),      # lstm_units
        random.choice([32, 64]),        # cnn_filters
        random.uniform(0.3, 0.6),       # dropout_rate
        random.choice([1e-4, 5e-4]),    # learning_rate
        random.choice([64, 128])        # dense_units
    ]

def mutate_individual(individual):
    idx = random.randint(0, len(individual)-1)
    if idx == 0:
        individual[idx] = random.choice([64, 128])
    elif idx == 1:
        individual[idx] = random.choice([32, 64])
    elif idx == 2:
        individual[idx] = random.uniform(0.3, 0.6)
    elif idx == 3:
        individual[idx] = random.choice([1e-4, 5e-4])
    elif idx == 4:
        individual[idx] = random.choice([64, 128])
    return individual,

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutate_individual)
toolbox.register("select", tools.selTournament, tournsize=3)

# Model Building
def build_model(params, embedding_matrix):
    lstm_units, cnn_filters, dropout_rate, lr, dense_units = params
    
    input_layer = Input(shape=(MAX_LEN,))
    x = Embedding(VOCAB_SIZE, EMBEDDING_DIM,
                  weights=[embedding_matrix],
                  trainable=False)(input_layer)
    x = SpatialDropout1D(0.3)(x)
    
    # LSTM Branch
    lstm = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    
    # CNN Branch
    conv = Conv1D(cnn_filters, 3, activation='relu')(lstm)
    conv = GlobalMaxPooling1D()(conv)
    
    # Combined Features
    x = Dense(dense_units, activation='relu',
              kernel_regularizer=regularizers.l2(0.01))(conv)
    x = Dropout(dropout_rate)(x)
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=input_layer, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Genetic Evaluation
def evaluate(individual, X_train, y_train, X_val, y_val):
    try:
        model = build_model(individual, embedding_matrix)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=BATCH_SIZE,
            verbose=0,
            callbacks=[EarlyStopping(patience=2, restore_best_weights=True)]
        )
        return (history.history['val_accuracy'][-1],)
    except:
        return (0.0,)

# Main Execution
if __name__ == "__main__":
    # Load and prepare data
    df = load_data('../Dataset/Sarcasm_Headlines_Dataset_v2.json')
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_headline'], df['is_sarcastic'], test_size=0.2, random_state=42
    )
    
    # Tokenization
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)
    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN)
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=MAX_LEN)
    
    # Prepare GloVe embeddings
    embeddings_index = {}
    with open(GLOVE_PATH, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')
    
    embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
    for word, i in tokenizer.word_index.items():
        if i < VOCAB_SIZE:
            embedding_matrix[i] = embeddings_index.get(word, np.random.normal(size=EMBEDDING_DIM))

    # Genetic Optimization
    toolbox.register("evaluate", evaluate, 
                    X_train=X_train_seq, y_train=y_train,
                    X_val=X_test_seq, y_val=y_test)
    
    population = toolbox.population(n=POPULATION_SIZE)
    algorithms.eaSimple(population, toolbox, cxpb=CXPB, mutpb=MUTPB,
                        ngen=GENERATIONS, verbose=True)
    
    # Get best individual
    best_individual = tools.selBest(population, 1)[0]
    print(f"Best Parameters: {best_individual}")
    
    # Train final model
    final_model = build_model(best_individual, embedding_matrix)
    history = final_model.fit(
        X_train_seq, y_train,
        validation_data=(X_test_seq, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
    )

    # Save results
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy Curves')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss Curves')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'))
    plt.close()

    y_pred = (final_model.predict(X_test_seq) > 0.5).astype(int)
    report = classification_report(y_test, y_pred)
    
    with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    plt.close()

    print(f"Results saved to {OUTPUT_DIR}")