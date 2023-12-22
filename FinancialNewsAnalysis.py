import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time


with st.spinner('Fetching Latest ML Model'):
    # Create the model, including its weights and the optimizer
    #model = tf.keras.models.load_model('\model\sentiment_model.h5')
    model = tf.keras.models.load_model('sentiment_model.h5')
    time.sleep(1)
    st.success('Model Loaded!')


st.title('Financial News Analysis App \n\n')
st.subheader('Input a sentence.') 


user_input=st.text_input("Sentense")
#Data cleansing before processing
user_input=user_input.lower() #We convert our texts to lowercase.
user_input=user_input.replace("[^\w\s]","") #We remove punctuation marks from our texts.
user_input=user_input.replace("\d+","") #We are removing numbers from our texts.
user_input=user_input.replace("\n","").replace("\r","") #We remove spaces in our texts.



def check_sentiment(user_input):
#if st.button("Check sentiment"):
    if(len(user_input)>0):
        # Tokenize and pad the user input
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts([user_input])
        sequences = tokenizer.texts_to_sequences([user_input])
        #padded_sequences = pad_sequences(sequences)
        padded_sequences = pad_sequences(sequences, maxlen=30, padding="post")

        # Make sentiment prediction
        prediction = model.predict(padded_sequences)[0, 0]

        # Display the result
        sentiment = "Positive" if prediction > 0.5 else "Negative"
        

        st.write("Sentiment: ", sentiment)
        st.write("Score: ", prediction)
       
check_sentiment(user_input)