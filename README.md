# NLP_Synergy

## Project Overview

NLP_Synergy is an end-to-end natural language processing project designed to perform key NLP tasks, including **part-of-speech tagging (POS)**, **named entity recognition (NER)**, and **text classification**. This project utilizes various NLP techniques such as Hidden Markov Models (HMMs), Stanford NER Tagger, TF-IDF, and Word2Vec embeddings, integrating syntactic parsing and statistical machine learning models.

## Tasks Covered

### 1. **Text Preprocessing**
- Sentence and token segmentation using NLTK.
- Stopwords removal and text normalization.

### 2. **Feature Extraction**
- **TF-IDF Vectors:** Computed using sklearn's `TfidfVectorizer`.
- **Word2Vec Embeddings:** Averaged word embeddings for sentence-level representations.

### 3. **Text Classification**
- Implemented and compared the performance of:
  - Naive Bayes
  - Logistic Regression

Classification was performed on movie summaries from the CMU corpus to categorize movies into genres: Thriller, Drama, Crime Fiction, and Romantic Comedy.

### 4. **Part-of-Speech Tagging**
- Implemented using Hidden Markov Models (HMM).
- Tagging based on NLTK's Treebank corpus.
- Evaluated using accuracy on a reserved test set.

### 5. **Named Entity Recognition (NER)**
- Utilized Stanford NER Tagger (version 4.2.0) for entity tagging.
- Extracted entities from movie summaries, categorizing them into predefined classes.

## Dataset
- CMU movie summary corpus.
- Genres classified: Thriller (0), Drama (1), Crime Fiction (2), Romantic Comedy (3).

## Performance Evaluation
- Metrics used: Precision, Recall, F1-score, Accuracy.
- Detailed performance comparison available in `classification_report.csv`.

## Requirements
- Python 3.10+
- Libraries:
  - numpy
  - pandas
  - scikit-learn
  - nltk
  - random

## Project Structure
- `classification.py`: Implementation of classification models and training.
- `parser.py`: Text preprocessing and feature extraction functions.
- `ner.py`: Implementation of Named Entity Recognition using Stanford NER.
- `classification_report.csv`: Performance evaluation of models.
- Prediction CSVs: Model predictions on the test dataset.

## Usage
- Install dependencies:
```bash
pip install numpy pandas scikit-learn nltk
```
- Ensure Stanford NER (v4.2.0) and Java (v1.8+) are installed.
- Execute Python scripts via command line:
```bash
python classification.py
python ner.py
```

## Authors
- Aashik Mathew Prosper

