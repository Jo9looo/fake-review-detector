# -*- coding: utf-8 -*-
import re
import string
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Downloader helper
nltk.download('stopwords', quiet=True)

# Initialize
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
analyzer = SentimentIntensityAnalyzer()

def preprocess_text(text):
    """
    Cleans text: lowercases, removes digits and punctuation, removes stopwords, and stems words.
    """
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 1]
    return ' '.join(tokens)

def extract_features(text):
    """
    Extracts 5 linguistic features from raw text:
    1. text_length
    2. exclamation_count
    3. uppercase_ratio
    4. sentiment_score
    5. lexical_diversity
    """
    words = text.split()
    word_count = len(words) if words else 1
    
    text_length = len(text)
    exclamation_count = text.count('!')
    
    upper_words = [w for w in words if w.isupper() and len(w) > 1]
    uppercase_ratio = len(upper_words) / word_count
    
    sentiment_score = analyzer.polarity_scores(text)['compound']
    
    unique_words = set(w.lower() for w in words)
    lexical_diversity = len(unique_words) / word_count
    
    # Return as DataFrame matching the scaler expectations
    return pd.DataFrame([[
        text_length, 
        exclamation_count, 
        uppercase_ratio, 
        sentiment_score, 
        lexical_diversity
    ]], columns=['text_length', 'exclamation_count', 'uppercase_ratio', 'sentiment_score', 'lexical_diversity'])
