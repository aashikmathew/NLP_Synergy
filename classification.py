# CS421: Natural Language Processing
# University of Illinois at Chicago
# Spring 2025
# Project Part 1
#
# Do not rename/delete any functions or global variables provided in this template and write your solution
# in the specified sections. Use the main function to test your code when running it from a terminal.
# Avoid writing that code in the global scope; however, you should write additional functions/classes
# as needed in the global scope. These templates may also contain important information and/or examples
# in comments so please read them carefully. If you want to use external packages not specified in the
# assignment then you need prior approval from course staff.
#
# This code will be graded automatically using Gradescope.
# =========================================================================================================
import pandas as pd
import numpy as np
import pickle as pkl
import nltk
import time
import csv
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Before running code that makes use of Word2Vec, you will need to download the provided w2v.pkl file
# which contains the pre-trained word2vec representations from Blackboard

# If you store the downloaded .pkl file in the same directory as this Python file, leave the global EMBEDDING_FILE variable below as is.  
# If you store the file elsewhere, you will need to update the file path accordingly.
EMBEDDING_FILE = "Word2Vec Model.pkl"

# Function: load_w2v
# filepath: path of w2v.pkl
# Returns: A dictionary containing words as keys and pre-trained word2vec representations as numpy arrays of shape (300,)
def load_w2v(filepath):
    with open(filepath, 'rb') as fin:
        return pkl.load(fin)


# Function: load_as_list(fname)
# fname: A string indicating a filename
# Returns: Two lists: one a list of document strings, and the other a list of integers
#
# This helper function reads in the specified, specially-formatted CSV file
# and returns a list of documents (movie summaries) and a list of categorical values (label).
def load_as_list(fname):
    df = pd.read_csv(fname)
    movie_summaries = df['summary'].tolist()
    labels = df['genre'].tolist()
    ids = df['id'].tolist()
    return movie_summaries, labels, ids

# Function: clean_text(text)
# text: A string containing the text to be processed
# Returns: Text converted to lowercase and stripped of stopwords
# Note: This function will be independently tested. It won't affect your classfication models.
# This function prepares the text by removing stopwords and converting all text to lowercase.
# This standardizes the raw data, focusing on the meaningful content essential for effective analysis.
def clean_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Filter out stopwords using NLTK's English stopwords list
    from nltk.corpus import stopwords
    try:
        stop_words = set(stopwords.words('english'))
    except:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
    tokens = get_tokens(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

# Function: get_sentences(text)
# text: A string containing the text to be processed
# Returns: A list of sentences extracted from the text
# Note: This function will be independently tested. It won't affect your classfication models.
# This function uses NLTK's sent_tokenize to split the text into individual sentences.
# This is useful for tasks that require analysis at the sentence level.
def get_sentences(text):
    from nltk.tokenize import sent_tokenize
    try:
        sentences = sent_tokenize(text)
    except:
        nltk.download('punkt')
        sentences = sent_tokenize(text)
    return sentences


# Function to convert a given string into a list of tokens
# Args:
#   inp_str: input string 
# Returns: token list, dtype: list of strings
def get_tokens(inp_str):
    # Check for existence of tokenizer, download if not found
    # Tokenize input string and return list of tokens
    # Check for existence of tokenizer, download if not found
    from nltk.tokenize import word_tokenize
    try:
        tokens = word_tokenize(inp_str)
    except:
        nltk.download('punkt')
        tokens = word_tokenize(inp_str)
    return tokens

# Function: vectorize_train.  See project statement for more details.
# training_documents: A list of strings
# Returns: An initialized TfidfVectorizer model, and a document-term matrix, dtype: scipy.sparse.csr.csr_matrix
def vectorize_train(training_documents):
    # Initialize TfidfVectorizer with a tokenizer
    vectorizer = TfidfVectorizer(tokenizer=get_tokens, lowercase=True)
    # Fit model to training documents and create document-term matrix
    tfidf_train = vectorizer.fit_transform(training_documents)
    return vectorizer, tfidf_train

# Function: w2v(word2vec, token)
# word2vec: The pretrained Word2Vec representations as dictionary
# token: A string containing a single token
# Returns: The Word2Vec embedding for that token, as a numpy array of size (300,)
#
# This function provides access to 300-dimensional Word2Vec representations
# pretrained on Google News.  If the specified token does not exist in the
# pretrained model, it should return a zero vector; otherwise, it returns the
# corresponding word vector from the word2vec dictionary.
def w2v(word2vec, token):
    # Initialize a zero vector
    # Check if token exists in word2vec dictionary and get corresponding vector
    # Initialize a zero vector
    zero_vector = np.zeros(300)
    # Check if token exists in word2vec dictionary and get corresponding vector
    if token in word2vec:
        return word2vec[token]
    return zero_vector

# Function: string2vec(word2vec, user_input)
# word2vec: The pretrained Word2Vec model
# user_input: A string of arbitrary length
# Returns: A 300-dimensional averaged Word2Vec embedding for that string
#
# This function preprocesses the input string, tokenizes it using get_tokens, extracts a word embedding for
# each token in the string, and averages across those embeddings to produce a
# single, averaged embedding for the entire input.
def string2vec(word2vec, user_input):
    # Tokenize input string
    # Accumulate word vectors for each token
    # Average the accumulated vectors
    # Tokenize input string
    tokens = get_tokens(user_input)
    if not tokens:
        return np.zeros(300)
    
    # Accumulate word vectors for each token
    embedding_sum = np.zeros(300)
    for token in tokens:
        embedding_sum += w2v(word2vec, token)
    
    # Average the accumulated vectors
    embedding_avg = embedding_sum / len(tokens)
    return embedding_avg


# Function: instantiate_models()
# This function does not take any input
# Returns: Three instantiated machine learning models
#
# This function instantiates the four imported machine learning models, and
# returns them for later downstream use.  You do not need to train the models
# in this function.
def instantiate_models():
    # Instantiate and return machine learning models
    nb = GaussianNB()
    logistic = LogisticRegression(random_state=18, multi_class='multinomial', solver='lbfgs')
    return nb, logistic


# Function: train_model_tfidf(model, tfidf_train, training_labels)
# model: An instantiated machine learning model
# tfidf_train: A document-term matrix built from the training data
# training_labels: A list of integers 
# Returns: A trained version of the input model
#
# This function trains an input machine learning model using TFIDF
# embeddings for the training documents.
def train_model_tfidf(model, tfidf_train, training_labels):
    # Convert sparse matrix to dense for Naive Bayes
    tfidf_dense = tfidf_train.toarray()
    # Fit model to training data using TFIDF matrix
    model.fit(tfidf_dense, training_labels)
    return model

# Function: train_model_w2v(model, word2vec, training_documents, training_labels)
# model: An instantiated machine learning model
# word2vec: A pretrained Word2Vec model
# training_data: A list of training documents
# training_labels: A list of integers 
# Returns: A trained version of the input model
#
# This function trains an input machine learning model using averaged Word2Vec
# embeddings for the training documents.
def train_model_w2v(model, word2vec, training_documents, training_labels):
    # Convert training documents to Word2Vec embeddings
    w2v_embeddings = []
    for doc in training_documents:
        w2v_embeddings.append(string2vec(word2vec, doc))
    
    # Convert to numpy array
    w2v_embeddings = np.array(w2v_embeddings)
    
    # Fit model to these embeddings
    model.fit(w2v_embeddings, training_labels)
    return model

# Function: test_model_tfidf(model, vectorizer, test_documents, test_labels)
# model: An instantiated machine learning model
# vectorizer: An initialized TfidfVectorizer model
# test_data: A list of test documents
# Returns:  A list of predictions
#
# This function tests an input machine learning model by extracting features
# for each preprocessed test document and then predicting an output label for
# that document.  
def test_model_tfidf(model, vectorizer, test_documents):
    # Transform test documents into TFIDF representations
    tfidf_test = vectorizer.transform(test_documents)
    # Convert to dense array for Naive Bayes
    tfidf_test_dense = tfidf_test.toarray()
    # Predict using the TFIDF model
    preds = model.predict(tfidf_test_dense)
    return preds


# Function: test_model_w2v(model, word2vec, training_documents, training_labels)
# model: An instantiated machine learning model
# word2vec: A pretrained Word2Vec model
# test_data: A list of test documents
# Returns: A list of predictions
#
# This function tests an input machine learning model by extracting features
# for each preprocessed test document and then predicting an output label for
# that document.  
def test_model_w2v(model, word2vec, test_documents):
    # Convert test documents to Word2Vec embeddings
    w2v_embeddings = []
    for doc in test_documents:
        w2v_embeddings.append(string2vec(word2vec, doc))
    
    # Convert to numpy array
    w2v_embeddings = np.array(w2v_embeddings)
    
    # Predict using the Word2Vec model
    preds = model.predict(w2v_embeddings)
    return preds


# Function: evaluate_predictions(test_labels, preds)
# test_labels: A list of actual labels for the test dataset
# preds: A list of predicted labels produced by the model
# Returns: Returns precision, recall, F1 score, and accuracy metrics
def evaluate_performance(test_labels, preds):
    precision = precision_score(test_labels, preds, average='weighted')
    recall = recall_score(test_labels, preds, average='weighted')
    f1 = f1_score(test_labels, preds, average='weighted')
    accuracy = accuracy_score(test_labels, preds)
    return precision, recall, f1, accuracy

# Use this main function to test your code. Sample code is provided to assist with the assignment;
# Some of the provided  code will help you in answering
# project questions, but it won't work correctly until all functions have been implemented.
if __name__ == "__main__":
    print("*************** Loading data & processing *****************")
    # Load the dataset
    print("Loading dataset.csv....")
    documents, labels, train_ids = load_as_list("dataset.csv")
    

    # Load the Word2Vec representations so that you can make use of it later
    print("Loading Word2Vec representations....")
    word2vec = load_w2v(EMBEDDING_FILE)

    # Compute TFIDF representations so that you can make use of them later
    print("Computing TFIDF representations....")
    vectorizer, tfidf_train = vectorize_train(documents)


    print("\n**************** Training models ***********************")
    # Instantiate and train the machine learning models
    print("Instantiating models....")
    nb_tfidf, logistic_tfidf = instantiate_models()
    nb_w2v, logistic_w2v = instantiate_models()

    print("Training Naive Bayes models....")
    start = time.time() # This will help you monitor training times (useful once training functions are implemented!)
    nb_tfidf = train_model_tfidf(nb_tfidf, tfidf_train, labels)
    end = time.time()
    print("Naive Bayes + TFIDF trained in {0} seconds".format(end - start))

    start = time.time()
    nb_w2v = train_model_w2v(nb_w2v, word2vec, documents, labels)
    end = time.time()
    print("Naive Bayes + w2v trained in {0} seconds".format(end - start))

    print("Training Logistic Regression models....")
    start = time.time()
    logistic_tfidf = train_model_tfidf(logistic_tfidf, tfidf_train, labels)
    end = time.time()
    print("Logistic Regression + TFIDF trained in {0} seconds".format(end - start))

    start = time.time()
    logistic_w2v = train_model_w2v(logistic_w2v, word2vec, documents, labels)
    end = time.time()
    print("Logistic Regression + w2v trained in {0} seconds".format(end - start))


    print("\n***************** Testing models ***************************")
    test_documents, test_labels, test_ids = load_as_list("test_data.csv")

    models_tfidf = [nb_tfidf, logistic_tfidf]
    models_w2v = [nb_w2v, logistic_w2v]
    model_names = ["Naive Bayes", "Logistic Regression"]

    outfile = open("classification_report.csv", "w", newline='\n')
    outfile_writer = csv.writer(outfile)
    outfile_writer.writerow(["Name", "Precision", "Recall", "F1", "Accuracy"])

    for i, model in enumerate(models_tfidf):
        # Testing TFIDF model
        print(f"Testing {model_names[i]} + TFIDF....")
        tfidf_preds = test_model_tfidf(model, vectorizer, test_documents)
        # Save predictions with IDs
        with open(f"{model_names[i]}_TFIDF_predictions.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ID', 'Predicted Label'])
            for id, pred in zip(test_ids, tfidf_preds):
                writer.writerow([id, pred])

        # Evaluate predictions
        p, r, f, a = evaluate_performance(test_labels, tfidf_preds)
        outfile_writer.writerow([model_names[i] + " + TFIDF", p, r, f, a])

        # Testing Word2Vec model
        print(f"Testing {model_names[i]} + w2v....")
        w2v_preds = test_model_w2v(models_w2v[i], word2vec, test_documents)
        # Save predictions with IDs
        with open(f"{model_names[i]}_w2v_predictions.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ID', 'Predicted Label'])
            for id, pred in zip(test_ids, w2v_preds):
                writer.writerow([id, pred])

        # Evaluate predictions
        p, r, f, a = evaluate_performance(test_labels, w2v_preds)
        outfile_writer.writerow([model_names[i] + " + w2v", p, r, f, a])

    outfile.close()
