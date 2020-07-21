from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app= Flask(__name__)

pickle_in= open('classifier.pkl','rb')
ml_model= pickle.load(pickle_in)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        try:
            age= float(request.form['age'])
            salary= float(request.form['salary'])
            pred_args=[age,salary]
            pred_args_arr= np.array(pred_args).reshape(1,-1)                    
            model_prediction= ml_model.predict(pred_args_arr)
            
        except ValueError:
            return "Please check values"
        
    return render_template('predict.html', prediction= model_prediction)    

if __name__ == '__main__':
    app.run()