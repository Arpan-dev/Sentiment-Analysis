import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
import re
import pandas as pd
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
nltk.download("stopwords")
sw=set(stopwords.words("english"))
neg_sw=["no","nor","not"]
pos_sw = [i for i in sw if i not in neg_sw]

def url_rmv(df):
    temp = []
    for i in df:
      #print(f"Data = {i} \n\n")
      a=re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", i)
      temp.append(a)
    return temp

def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    
    L=[]
    for i in text:
        if i.isalnum():
            L.append(i)
    
    text=L[:]          #copying list
    L.clear()
    
    for i in text:
        if i not in pos_sw and i not in string.punctuation:
            L.append(i)
    
    text=L[:]
    L.clear()
    
    for i in text:
        L.append(ps.stem(i))
    
    
    return " ".join(L)

cv = pickle.load(open('final_cv.pickle','rb'))
model = pickle.load(open('final_model.pickle','rb'))

st.title("Twitter Sentiment Analysis")

tweet = st.text_area("Enter the tweet")
st.write('The tweet is :---> ' ,  tweet)

tweet = [tweet]
if st.button('Predict'):

    # 1. Removing URL
    tweet = pd.DataFrame(tweet,columns = ["text"])
    url_rmvd_text = url_rmv(tweet["text"])
    tweet["url_rmvd_text"]= url_rmvd_text
    
    # 2. Preprocess
    tweet["trans_text"]=tweet["url_rmvd_text"].apply(transform_text)
    print(tweet["trans_text"])
    
    # 3. Vectorize
    X = cv.transform(tweet["trans_text"])

    # 4. Predict
    pred=model.predict(X)[0]
    
    # 5. Display
    if pred == 0:
        st.header("Negative Sentiment")
    elif pred == 1:
        st.header("Neutral Sentiment")
    else:
        st.header("Positive Sentiment")
