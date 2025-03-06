# CS421: Natural Language Processing
# University of Illinois at Chicago
# Spring 2025
# Project Part 1

# Do not rename/delete any functions or global variables provided in this template and write your solution
# in the specified sections. Use the main function to test your code when running it from a terminal.
# Avoid writing that code in the global scope; however, you should write additional functions/classes
# as needed in the global scope. These templates may also contain important information and/or examples
# in comments so please read them carefully.

import nltk
from nltk.corpus import treebank
import numpy as np
import random

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('treebank')

# Function: get_treebank_data
# Input: None
# Returns: Tuple (train_sents, test_sents)
#
# This function fetches tagged sentences from the NLTK Treebank corpus, calculates an index for an 80-20 train-test split,
# then splits the data into training and testing sets accordingly.

def get_treebank_data():
    # Fetch tagged sentences from the NLTK Treebank corpus.
    sentences = treebank.tagged_sents()
    # Calculate the split index for an 80-20 train-test split.
    split = int(len(sentences) * 0.8)
    # Split the data into training and testing sets.
    train_sents = sentences[:split]
    test_sents = sentences[split:]
    return train_sents, test_sents


# Function: compute_tag_trans_probs
# Input: train_data (list of tagged sentences)
# Returns: Dictionary A of tag transition probabilities
#
# Iterates over training data to compute the probability of tag bigrams (transitions from one tag to another).

def compute_tag_trans_probs(train_data):
    tag_bigrams = {}
    tag_counts = {}
    # Iterate through each sentence to count tag transitions.
    for sent in train_data:
        prev_tag = "<s>"  # Start symbol for each sentence
        for word, tag in sent:
            if prev_tag not in tag_bigrams:
                tag_bigrams[prev_tag] = {}
            tag_bigrams[prev_tag][tag] = tag_bigrams[prev_tag].get(tag, 0) + 1

            tag_counts[prev_tag] = tag_counts.get(prev_tag, 0) + 1
            prev_tag = tag
        # At end of sentence, count transition to end symbol </s>
        if prev_tag not in tag_bigrams:
            tag_bigrams[prev_tag] = {}
        tag_bigrams[prev_tag]["</s>"] = tag_bigrams[prev_tag].get("</s>", 0) + 1
        tag_counts[prev_tag] = tag_counts.get(prev_tag, 0) + 1

    # Convert counts to probabilities.
    A = {}
    for prev_tag, transitions in tag_bigrams.items():
        A[prev_tag] = {}
        for curr_tag, count in transitions.items():
            A[prev_tag][curr_tag] = count / tag_counts[prev_tag]
    return A

# Function: compute_emission_probs
# Input: train_data (list of tagged sentences)
# Returns: Dictionary B of tag-to-word emission probabilities
#
# Iterates through each sentence in the training data to count occurrences of each tag emitting a specific word, then calculates probabilities.

def compute_emission_probs(train_data):
    emission_counts = {}
    tag_counts = {}
    # Count each word and tag pair and count tags.
    for sent in train_data:
        for word, tag in sent:
            if tag not in emission_counts:
                emission_counts[tag] = {}
            emission_counts[tag][word] = emission_counts[tag].get(word, 0) + 1
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    # Convert counts to probabilities.
    B = {}
    for tag, word_counts in emission_counts.items():
        B[tag] = {}
        for word, count in word_counts.items():
            B[tag][word] = count / tag_counts[tag]
    return B

# Function: viterbi_algorithm
# Input: words (list of words that have to be tagged), A (transition probabilities), B (emission probabilities)
# Returns: List (the most likely sequence of tags for the input words)
#
# Implements the Viterbi algorithm to determine the most likely tag path for a given sequence of words, using given transition and emission probabilities.

def viterbi_algorithm(words, A, B):
    states = list(B.keys())
    Vit = [{}]  # Viterbi table
    path = {}

    # Initialization for t = 0 using the start symbol <s>
    for state in states:
        # Multiply the transition probability from "<s>" to state by the emission probability of the first word.
        if "<s>" in A and state in A["<s>"]:
            Vit[0][state] = A["<s>"][state] * B.get(state, {}).get(words[0], 0.0001)
        else:
            Vit[0][state] = 0.0001 * B.get(state, {}).get(words[0], 0.0001)
        path[state] = [state]

    # Recursion: iterate for each word t > 0.
    for t in range(1, len(words)):
        Vit.append({})
        newpath = {}
        for curr_state in states:
            (prob, best_prev) = max(
                ((Vit[t-1][prev_state] *
                  A.get(prev_state, {}).get(curr_state, 0.0001) *
                  B.get(curr_state, {}).get(words[t], 0.0001), prev_state)
                 for prev_state in states),
                key=lambda x: x[0]
            )
            Vit[t][curr_state] = prob
            newpath[curr_state] = path[best_prev] + [curr_state]
        path = newpath

    # Termination: find the best final state that transitions to the end symbol </s>
    (prob, best_last) = max(
        ((Vit[len(words)-1][state] * A.get(state, {}).get("</s>", 0.0001), state)
         for state in states),
        key=lambda x: x[0]
    )
    return path[best_last]

# Function: evaluate_pos_tagger
# Input: test_data (tagged sentences for testing), A (transition probabilities), B (emission probabilities)
# Returns: Float (accuracy of the POS tagger on the test data)
#
# Evaluates the POS tagger's accuracy on a test set by comparing predicted tags to actual tags and calculating the percentage of correct predictions.

def evaluate_pos_tagger(test_data, A, B):
    correct = 0
    total = 0
    for sent in test_data:
        words = [word for word, tag in sent]
        actual_tags = [tag for word, tag in sent]
        predicted_tags = viterbi_algorithm(words, A, B)
        for atag, ptag in zip(actual_tags, predicted_tags):
            if atag == ptag:
                correct += 1
            total += 1
    accuracy = correct / total if total > 0 else 0
    return accuracy

# Main function to train and evaluate the POS tagger.
if __name__ == "__main__":
    train_sents, test_sents = get_treebank_data()
    A = compute_tag_trans_probs(train_sents)
    B = compute_emission_probs(train_sents)

    # Print specific probabilities
    print(f"P(VB -> DT): {A.get('VB', {}).get('DT', 0):.4f}")  # Expected around 0.2296
    print(f"P(DT -> 'the'): {B.get('DT', {}).get('the', 0):.4f}")  # Expected around 0.4986

    # Evaluate the model's accuracy
    accuracy = evaluate_pos_tagger(test_sents, A, B)
    print(f"Accuracy of the HMM-based POS Tagger: {accuracy:.4f}")  # Expected around 0.8743
