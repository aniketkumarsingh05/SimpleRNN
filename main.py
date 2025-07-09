import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Mapping of word index bacl to words(for understanding)
word_index =imdb.get_word_index()
reverse_word_index = {value : key for key, value in word_index.items()}

# Load the pre- trained model Relu activation

model =load_model('simple_rnn_imdb.h5')

# Helper Functions
# Function to decode review

def decode_review(encoded_review):


    decode_review = ' '.join([reverse_word_index.get(i - 3,'?') for i in encoded_review])

# Function to preprocess user input

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review



import streamlit as st
# Streamlit app
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it a positive or negative")

# User Input
user_input = st.text_area('Moview Review')

if st.button('Classify'):
    preprocess_input = preprocess_text(user_input)

    # Make prediction
    prediction = model.predict(preprocess_input)
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'Negative'
    

    # Display the result
    st.write(f'Sentiment : {sentiment}')
    st.write(f'Prediction Score : {prediction[0][0]}')
else:
    st.write("Please enter a movie review.")    

