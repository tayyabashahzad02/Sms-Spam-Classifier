import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()



def clean_message(message):
    message = message.lower()
    message = nltk.word_tokenize(message)

    z = []
    for i in message:
        if i.isalnum():
            z.append(i)

    message = z[:]
    z.clear()

    for i in message:
        if i not in stopwords.words('english') and i not in string.punctuation:
            z.append(i)

    message = z[:]
    z.clear()

    for i in message:
        z.append(ps.stem(i))

    return " ".join(z)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("SMS Spam Classifier")

input_sms = st.text_input("Enter the Message")

#preprocess
cleaned_message = clean_message(input_sms)

#vectorize
vector_input = tfidf.transform([cleaned_message])

#predict
result = model.predict(vector_input)[0]

#Display
if result == 1:
    st.header("Spam")
else:
    st.header("Not Spam")