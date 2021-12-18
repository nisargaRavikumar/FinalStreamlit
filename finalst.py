#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import streamlit as st
import glob
import os
import seaborn as sns
from matplotlib import pyplot as plt
import joblib
import plotly.express as px
#from matplotlib import colors as mcolors
import base64


st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url('https://media.istockphoto.com/vectors/books-seamless-pattern-vector-id470721440?k=20&m=470721440&s=612x612&w=0&h=o3yXdl1xi_kbMdyP6NRirKX4Y0m8WjVgul9zl4i5LtU=')
    }}
    .css-1d391kg {{
        background-color : gainsboro
    }}
    .stTextInput, .stSelectbox {{
        font-weight : bold
    }}
    </style>
    """,
    unsafe_allow_html=True

)
add_selectbox = st.sidebar.radio(
    "Select the option",
    ("Book Recommendations", "Inventory Forecasting")
)

 
def bookrecomm():
    st.title("Books Recommendations for User")
    st.write("-------------------------------------------------------------------------------------------------")
    lmodel = keras.models.load_model("my_model")
    def recommend(user_id):
        books = pd.read_csv('books_cleaned.csv')
        ratings = pd.read_csv('ratings.csv')
  
        book_id = list(ratings.book_id.unique()) #grabbing all the unique books
  
        book_arr = np.array(book_id) #geting all book IDs and storing them in the form of an array
        user_arr = np.array([user_id for i in range(len(book_id))])
        prediction = lmodel.predict([book_arr, user_arr])
  
        prediction = prediction.reshape(-1) #reshape to single dimension
        prediction_ids = np.argsort(-prediction)[0:5]

        recommended_books = pd.DataFrame(books.iloc[prediction_ids], columns = ['book_id', 'isbn', 'authors', 'title', 'average_rating' ])
        print('Top 5 recommended books for you: \n')
        return recommended_books
      
    user = st.text_input('Enter a user ID',value="0")
    user = int(user)
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        st.dataframe(recommend(user).style.set_properties(**{'background-color': 'aliceblue'}))

def salesforecast():
    st.title("Inventory Forecasting")
    st.write("-------------------------------------------------------------------------------------------------")
    
    model = joblib.load('finalized_model1.sav')
    X_test= pd.read_csv(r'X_test.csv')
    test= pd.read_csv(r'IFDtest.csv',parse_dates=['Transaction_date'])
    book_ids=test['book_id'].unique()
    result = model.predict(X_test, num_iteration=model.best_iteration)
    forecast = pd.DataFrame({"date":test["Transaction_date"],
                        "store":test["store_id"],
                        "item":test["book_id"],
                        "sales":result
                        })      

    Store = st.selectbox('Enter Store ID',(1,2,3,4,5,6,7,8,9,10))
    Item = st.selectbox('Enter Book ID',tuple(book_ids)) 
    if st.button("Predict"): 
        df = forecast[(forecast.store == Store) & (forecast.item == Item)]                        
        fig = px.line(        
            df, #Data Frame
            x = "date", #Columns from the data frame
            y = "sales",
            title = "Line frame"
        )
        st.write(fig)



    
               
if add_selectbox == 'Book Recommendations':
    bookrecomm()   
elif add_selectbox == 'Inventory Forecasting':
    salesforecast()   

