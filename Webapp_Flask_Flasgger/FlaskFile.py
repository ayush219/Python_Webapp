from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

pickle_in= open('classifier.pkl','rb')
classifier=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return 'welcome world'

@app.route('/predict')
def predict():
    age=request.args.get('age')
    salary=request.args.get('salary')
    prediction=classifier.predict([[age,salary]])
    return "Predicted value is" + str(prediction)

@app.route('/predict_file',methods=['POST'])
def predict_file():
    df_test=pd.read_csv(request.files.get('file'), header=None)    
    prediction=classifier.predict(df_test)
    return "Predicted value for file is" + str(prediction)

@app.route('/predict_user',methods=['GET'])
def predict_user():
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
    age=request.args.get('age')
    salary=request.args.get('salary')
    prediction=classifier.predict([[age,salary]])
    return "Predicted value is" + str(prediction)

@app.route('/predict_file_user',methods=['POST'])
def predict_file_user():
    """
    Please enter the values
    ---
    parameters:
        - name: file
          in: formData
          type: file
          required: true
    responses:
        200:
            description: Output values

    """
    df_test=pd.read_csv(request.files.get('file'), header=None)    
    prediction=classifier.predict(df_test)
    return "Predicted value for file is" + str(prediction)

if __name__== '__main__':
    app.run()