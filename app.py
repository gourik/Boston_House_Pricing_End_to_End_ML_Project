import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__) #initial point of application from where it runs 
#define app:
reg_model=pickle.load(open('regression_model.pkl','rb'))
sc_model=pickle.load(open('scaling.pkl','rb'))
@app.route('/') #this redirects to home page by default:
def home():
    return render_template('home.html')
#To predict new data from web page:
@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    #'data' is in dictionary format..but we need values.To get single list of values we have to convert it into 
    #list. As we had two dimensional data for model prediction, we have to convert this single dimensional list into 2D.
    #But, first we need to convert it into array, and then convert it using reshape(1,-1.) 
    print(np.array(list(data.values())).reshape(1,-1)) 
    new_data=sc_model.transform(np.array(list(data.values())).reshape(1,-1))
    output=reg_model.predict(new_data)    
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input=sc_model.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=reg_model.predict(final_input)[0]
    return render_template('home.html',prediction_text="The house price prediction is{}".format(output))


if __name__=="__main__":
    app.run(debug=True)