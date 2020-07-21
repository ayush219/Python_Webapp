import pandas as pd
import numpy as np
import pickle
import streamlit as st

pickle_in= open('classifier.pkl','rb')
classifier=pickle.load(pickle_in)

def predict_user(age, salary):
    """
    Please enter the values
    ---
    parameters:
        - name: age
          in: query
          type: number
          requiered: true
        - name: salary
          in: query
          type: number
          required: true
    responses:
        200:
            description: Output values
    """
    
    prediction=classifier.predict([[age,salary]])
    return "Predicted value is" + str(prediction)

def main():
    st.title("Prediction")
    html_temp="""
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;"> Prediction Model</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    age= st.text_input("age","Type here")
    salary= st.text_input("salary","Type here")
    result=" "
    if st.button("Predict"):
        result=predict_user(age,salary)
    st.success("Ouput is {}".format(result))
    if st.button("About"):
        st.text("Built with Streamlit")

if __name__== '__main__':
    main()