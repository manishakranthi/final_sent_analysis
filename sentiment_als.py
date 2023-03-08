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


if __name__ == '__main__':
    st.title('Financial Sentiment Analysis :bar_chart:')
    st.write('A simple sentiment analysis classification app')
    st.subheader('Give your input below')
    sentence = st.text_area('Enter your text here',height=200)
    predict_btt = st.button('predict')
    loaded_model = pickle.load(open('sentiment_analysis.p', 'rb')) 
if predict_btt:
    disp=" "
	
a = loaded_model.predict([sentence])[0]
if(a== 1):
	disp = "positive sentiment"
elif(a==0):
	disp = "neutral sentiment"
else:
	disp = "negative sentiment"
st.write('The sentiment of the given text is:', disp)

        
