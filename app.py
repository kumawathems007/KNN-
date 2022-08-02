import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
model = pickle.load(open('kkn.pkl', 'rb')) 

@app.route('/')
def home():
  
    return render_template("index2.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    
    
    '''
    For rendering results on HTML GUI
    '''
    Age = float(request.args.get('Age'))
    SibSp = float(request.args.get('SibSp'))
    Parch = float(request.args.get('Parch'))
    Fare = float(request.args.get('Fare'))
    Gender = float(request.args.get('Gender'))
    Pclass = float(request.args.get('Pclass'))
    
    prediction = model.predict([[Age, SibSp,Parch,Fare,Gender,Pclass]])
     
        
    return render_template('index2.html', prediction_text='KNN Model  has predicted Survived for given Data is : {}'.format(prediction))
   
if __name__ == "__main__":
    app.run(debug=True)
