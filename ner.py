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
import nltk
from nltk.tag.stanford import StanfordNERTagger


# Function: get_ner(text)
# fname: A string containing text to be processed
# Returns: A list of tuples containing (token, tag)
def get_ner(text, path_to_jar, path_to_model):
    # Initialize Stanford NER tagger with the provided paths
    tagger = StanfordNERTagger(
        path_to_model,
        path_to_jar,
        encoding='utf-8'
    )
    
    # Tokenize the input text into words
    words = nltk.word_tokenize(text)
    
    # Apply the NER tagger to get tagged tokens
    tagged_word_list = tagger.tag(words)
    
    # Filter out tokens with 'O' tag (which means "Outside" or not a named entity)
    # Only keep actual named entities
    named_entities = [(word, tag) for word, tag in tagged_word_list if tag != 'O']
    
    return named_entities
    
# Use this main function to test your code. Sample code is provided to assist with the assignment;
# feel free to change/remove it. Some of the provided sample code will help you in answering
# questions, but it won't work correctly until all functions have been implemented.
if __name__ == "__main__":
    
    # Paths to Stanford NER tagger model and jar file.
    path_to_jar = "stanford_ner/stanford-ner.jar"
    path_to_model = "stanford_ner/classifiers/english.all.3class.distsim.crf.ser.gz"
    data = pd.read_csv('dataset.csv')

    
    for index, row in data.iterrows():
        if index >= 2: 
            break

        title = row['title']
        summary = row['summary']
        entities = get_ner(summary,path_to_jar,path_to_model)

        print(f"Movie Title: {title}")
        print(f"List of Named Entities: {entities}")


### Expected Outputs

# Movie Title: Incubus
# List of Named Entities: [('Orin', 'PERSON'), ('Kiefer', 'PERSON'), ('Peter', 'PERSON'), ('Peter', 'PERSON'), ('Jay', 'PERSON'), ('Holly', 'PERSON'), ('Peter', 'PERSON'), ('Holly', 'PERSON'), ('Jay', 'PERSON'), ('Jay', 'PERSON'), ('Peter', 'PERSON'), ('Jay', 'PERSON'), ('Jay', 'PERSON'), ('Jay', 'PERSON')]
# Movie Title: H
# List of Named Entities: [('Shin', 'PERSON'), ('Hyun', 'PERSON'), ('Shin', 'PERSON'), ('Shin', 'PERSON'), ('Chu', 'PERSON'), ('Kang', 'ORGANIZATION'), ('Shin', 'PERSON')]
# Movie Title: Bring Me the Head of Alfredo Garcia
# List of Named Entities: [('Alfredo', 'PERSON'), ('Garcia', 'PERSON'), ('El', 'ORGANIZATION'), ('Jefe', 'ORGANIZATION'), ('El', 'ORGANIZATION'), ('Jefe', 'ORGANIZATION'), ('$', 'MONEY'), ('1', 'MONEY'), ('million', 'MONEY'), ('Alfredo', 'PERSON'), ('Garcia', 'PERSON'), ('Mexico', 'LOCATION'), ('City', 'LOCATION'), ('El', 'ORGANIZATION'), ('Jefe', 'ORGANIZATION'), ('Quill', 'PERSON'), ('Bennie', 'PERSON'), ('United', 'ORGANIZATION'), ('States', 'ORGANIZATION'), ('Army', 'ORGANIZATION'), ('Garcia', 'PERSON'), ('Garcia', 'LOCATION'), ('Elita', 'LOCATION'), ('Garcia', 'PERSON'), ('Bennie', 'PERSON'), 
# ('Garcia', 'PERSON'), ('Quill', 'PERSON'), ('Max', 'PERSON'), ('$', 'MONEY'), ('10,000', 'MONEY'), ('Garcia', 'PERSON'), ('Garcia', 'PERSON'), ('Garcia', 'PERSON'), ('Bennie', 'PERSON'), ('Bennie', 'PERSON'), ('Bennie', 'PERSON'), ('Elita', 'PERSON'), ('Elita', 'LOCATION'), ('Bennie', 'PERSON'), ('Kristofferson', 'PERSON'), ('Elita', 'ORGANIZATION'), ('Elita', 'ORGANIZATION'), ('Elita', 'LOCATION'), ('Garcia', 'PERSON'), ('Mexico', 'LOCATION'), ('City', 'LOCATION'), ('Garcia', 'PERSON'), ('Garcia', 'PERSON'), ('Bennie', 'PERSON'), ('Elita', 'PERSON'), ('Garcia', 'PERSON'), ('Garcia', 'PERSON'), ('Garcia', 'PERSON'), ('Alfredo', 'PERSON'), 
# ('Elita', 'ORGANIZATION'), ('Garcia', 'PERSON'), ('Bennie', 'PERSON'), ('Quill', 'PERSON'), ('Garcia', 'PERSON'), ('Quill', 'ORGANIZATION'), ('Bennie', 'PERSON'), ('Bennie', 'PERSON'), ('Garcia', 'PERSON'), ('Mexico', 'LOCATION'), ('City', 'LOCATION'), ('Garcia', 'PERSON'), ('Bennie', 'PERSON'), ('Garcia', 'PERSON'), ('$', 'MONEY'), ('10,000', 'MONEY'), ('Elita', 'ORGANIZATION'), ('Bennie', 'PERSON'), ('El', 'ORGANIZATION'), ('Jefe', 'ORGANIZATION'), ('El', 'ORGANIZATION'), ('Jefe', 'ORGANIZATION'), ('Garcia', 'PERSON'), ('Bennie', 'PERSON'), ('Garcia', 'PERSON'), ('Elita', 'ORGANIZATION'), ('Bennie', 'PERSON'), ('El', 'ORGANIZATION'), ('Jefe', 'ORGANIZATION'), ('Bennie', 'PERSON'), ('El', 'ORGANIZATION'), ('Jefe', 'ORGANIZATION'), ('Bennie', 'PERSON'), ('Bennie', 'PERSON'), ('obliges', 'PERSON'), ('El', 'ORGANIZATION'), ('Jefe', 'ORGANIZATION')]