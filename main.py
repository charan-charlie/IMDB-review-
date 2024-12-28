# Step 1 : import libraries and load the model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

import streamlit as st

word_index = imdb.get_word_index()

reverse_word = {value : key for key ,value in word_index.items()}

# Load the pre-traied model with ReLU activation

model = load_model("Simple_rnn_imdb.h5")

def decode_review(encoded_view):
    return ' '.join([reverse_word.get(i-3," ? ") for i in encoded_view])

#Function to preprocess user input

def preprocess_text(text):

    words = text.lower().split()

    encoded_review = [word_index.get(word,2)+3 for word in words]
    padded_review = pad_sequences([encoded_review],padding = "pre",maxlen = 500)
    # maxlen is the parameter which is used for fixed input size 

    return padded_review

## streamlit app

st.title('IMDB Movie review sentiment Analysis')

st.write("Enter a movie review to classify it as positive or negative.")

user_input = st.text_area('Movie Review')

if st.button("Classify"):

    preprocess_input = preprocess_text(user_input)

    ## Make Prediction 
    prediction = model.predict(preprocess_input)

    sentiment = "Positive" if prediction[0][0] > 0.5 else 'Negative'

    #Display the result

    st.write(f"Sentiment : {sentiment}")
    st.write(f"Prediction Score : {prediction[0][0]}")
else:

    st.write("Please enter a movie review")

