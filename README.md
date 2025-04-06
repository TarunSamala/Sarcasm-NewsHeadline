# Sarcasm Aware: Sarcasm Detection in Headlines 🔍

![Task](https://img.shields.io/badge/Task-Text_Classification-blue) 
![Models](https://img.shields.io/badge/Models-Bi--LSTM,_Bi--GRU,_RandomForest,_BERT-orange) 
![Embeddings](https://img.shields.io/badge/Embeddings-GloVe,_Word2Vec,_FastText-green)

A machine learning project to classify sarcastic headlines using both traditional and deep learning models. Built to compare performance across architectures and embeddings.

## Project Overview 📋
This project detects sarcasm in news headlines using Kaggle's dataset. Implements **Bi-LSTM**, **Bi-GRU**, **RandomForest**, and **BERT** models with pretrained embeddings (**GloVe**, **Word2Vec**, **FastText**). Benchmarks performance across different architectures.

## Features ✨
- Multiple model architectures for comparison
- Pretrained embeddings support
- Detailed performance metrics
- BERT fine-tuning implementation
- Comprehensive evaluation framework

## Dataset 📂
**Source**: [Sarcasm Headlines Dataset](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection)
- **28,000+** news headlines
- Binary labels (sarcastic/not sarcastic)
- Balanced class distribution

## Models & Embeddings 🤖
### Implemented Models
| Model        | Type          | Implementation Details |
|--------------|---------------|------------------------|
| Bi-LSTM      | Deep Learning | 2-layer bidirectional  |
| Bi-GRU       | Deep Learning | With attention         |
| RandomForest | Traditional   | TF-IDF features        |
| BERT         | Transformer   | Fine-tuned base model  |

### Embeddings Used
- **GloVe**: `glove.6B.100d` (100D)
- **Word2Vec**: Google News 300D
- **FastText**: `crawl-300d-2M` (300D)

## Installation ⚙️
1. Clone repository:
```bash
git clone https://github.com/TarunSamala/Sarcasm-NewsHeadline.git