import pandas as pd
import numpy as np
import pickle
import streamlit as st
import nltk
nltk.download ("stopwords")
nltk.download("punkt")
nltk.download('wordnet')
nltk.download('omw-1.4')
import re
import string
from textblob import Word
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


loaded_model = pickle.load(open('filename', 'rb'))

def text_cleaning(line_from_column):
    text = line_from_column.lower()
    # Replacing the digits/numbers
    text = text.replace('d', '')
    # remove stopwords
    words = [w for w in text if w not in stopwords.words("english")]
    # apply stemming
    words = [Word(w).lemmatize() for w in words]
    # merge words 
    words = ' '.join(words)
    return text 

def load():
	''' Load the calculated TFIDF weights'''

	df = None
	with open('tfidf.pickle', 'rb') as f:
		df = pickle.load(f)
	return df 

if __name__ == '__main__':
    st.title('Financial Sentiment Analysis :bar_chart:')
    st.write('A simple sentiment analysis classification app')
    st.subheader('Input the Statment below')
    sentence = st.text_area('Enter your text here',height=200)
    predict_btt = st.button('predict')
    loaded_model = pickle.load(open('filename', 'rb')) 
    if predict_btt:
        clean_text = []
        i = text_cleaning(sentence)
        clean_text.append(i)
        data = load(clean_text)

        # st.info(data)
        prediction = loaded_model.predict(data)

        prediction_prob_negative = prediction[0][-1]
        prediction_prob_neutral = prediction[0][0]
        prediction_prob_positive= prediction[0][1]

        prediction_class = prediction(axis=-1)[0]
        print(prediction)
        st.header('Prediction using SVC model')
        if prediction_class == -1:
          st.warning('Sentence has negative sentiment')
        if prediction_class == 0:
          st.success('Sentence has neutral sentiment')
        if prediction_class==1:
          st.success('Sentence has positive sentiment')
