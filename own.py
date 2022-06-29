import streamlit as st
import joblib
import pandas as pd
import numpy as np
st.title("**_REAL-FAKE CLASSIFYER_**")
#st.subheader("Enter Message")
ip=st.text_area('Text to CLASSIFY')
option=st.sidebar.selectbox('select the model',('svm','svm+pipeline','multinominalNB','multinominalNB+pipeline'))
df=pd.read_csv("https://raw.githubusercontent.com/saiteja-99/data_set/main/news.csv")
st.sidebar.write("Use below Data to test")
st.sidebar.write(df.iloc[:,[2,3]])
if st.button('predict'):
    if option=='svm':
        model=joblib.load("svc")
        vect=joblib.load("vectorizer")
        op=np.array([ip]) 
        op=vect.transform(op)
        op=model.predict(op)
    elif option=='multinominalNB':
        model=joblib.load("multinominalNB")
        vect=joblib.load("vectorizer")
        op=np.array([ip]) 
        op=vect.transform(op)
        op=model.predict(op)
    elif option=='svm+pipeline':
        model=joblib.load("svcpipeline")
        op=model.predict([ip])
    else:
        model=joblib.load("multinominalNBpipeline")
        op=model.predict([ip])
    st.title(op[0])
    #st.balloons()
    st.snow()
