# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import pickle
import os

app = Flask(__name__)
model = pickle.load(open("PCA_model.pkl", "rb"))
@app.route('/')
def home():
    return render_template("pca.html")

@app.route('/result.html',methods=["POST","GET"])

def predict():
    f1 = request.form["power"]
    f2 = request.form["value"]
    f3 = request.form["reading1"]
    f4 = request.form["reading2"]
    f5 = request.form["reading3"]
   
    input_features = [[float(f1), float(f2), float(f3), float(f4), float(f5)]]
    #features_value = [np.array(input_features)]
    
    #features_name = ['Global_reactive_power', 'Global_intensity',
    #                 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'Sub_metering_4', 'Sub_metering_5']
    #df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(input_features)
    
    return render_template('result.html', prediction_text=output)

if __name__ =="__main__":
    app.run(debug=False)
    