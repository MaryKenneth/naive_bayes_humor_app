import numpy as np
import pandas as pd
import re

import spacy

from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import nltk
nltk.download('punkt')  # Download the punkt tokenizer data
nltk.download('wordnet')  # Download the WordNet data
nltk.download('averaged_perceptron_tagger')

nlp = spacy.load('en_core_web_sm')


def lemmatize_text(text):
    # Tokenize the text into words
    words = word_tokenize(text)
    
    # Initialize the WordNet lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Lemmatize each word in the list
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    
    # Join the lemmatized words into a single string
    lemmatized_text = ' '.join(lemmatized_words)
    
    return lemmatized_text

def tokens(x):
    return x.split()

def remove_common(text):
    pattern = r'the | the | a | to | is | an |”|“'
    text = text.lower()
    #clean_text = tokens(text)
    clean_text = re.sub(pattern," ",text)
    return clean_text

def lemmatize_text_with_pos(text):
    text = text.lower()
    words = word_tokenize(text)
    
    # Perform POS tagging
    pos_tags = pos_tag(words)
    
    # Initialize the WordNet lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Lemmatize each word based on its POS tag
    lemmatized_words = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(pos_tag))
                        for word, pos_tag in pos_tags]
    
    # Join the lemmatized words into a single string
    lemmatized_text = ' '.join(lemmatized_words)
    
    return lemmatized_text

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return 'a'  # Adjective
    elif treebank_tag.startswith('V'):
        return 'v'  # Verb
    elif treebank_tag.startswith('N'):
        return 'n'  # Noun
    elif treebank_tag.startswith('R'):
        return 'r'  # Adverb
    else:
        return 'n'  # Default to 'n' (Noun) if the POS tag is not recognized

def stemming(text):
    ps = PorterStemmer()
    text = text.lower()
    words = word_tokenize(text) 
    stem_word = [ps.stem(word) for word in words]

    # Join the lemmatized words into a single string
    stem_text = ' '.join(stem_word)
    return stem_text

#### Spacy Library
def lemmatize_spacy(text):
    doc = nlp(text)
    lemms = []
    for token in doc:
        lemms.append(token.lemma_)
    return lemms

def tokenise_spacy(text):
    doc = nlp(text)
    tokens = []  # Create an empty list to store tokens
    for token in doc:
        tokens.append(token.text)
    return tokens  # Return the list of tokens after the loop

#text = "She is a small thief and sadden stealing her coin"
#print(stemming(text))
#print(tokens(text))