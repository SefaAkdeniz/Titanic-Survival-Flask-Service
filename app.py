import pandas as pd
from flask import Flask,jsonify,request
import joblib

scaler = joblib.load('scaler.pkl')
model = joblib.load('model.pkl')

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

@app.route('/predict', methods=['POST'])
def index():
    
    body = request.json
    Pclass = body["Pclass"]
    Sex = body["Sex"]
    Age = body["Age"]
    SibSp = body["SibSp"]
    Parch = body["Parch"]
    Fare = body["Fare"]
    Embarked = body["Embarked"]
    
    try:     
        if Embarked == "Q":
            array=pd.DataFrame([[Pclass,Sex,Age,SibSp,Parch,Fare,0,1,0]])
        elif Embarked == "C":
            array=pd.DataFrame([[Pclass,Sex,Age,SibSp,Parch,Fare,1,0,0]])
        elif Embarked == "S":
            array=pd.DataFrame([[Pclass,Sex,Age,SibSp,Parch,Fare,0,0,1]])
            
        array = scaler.transform(array)    
        print(model.predict(array)[0])    
        print(type(model.predict(array)[0]))          
        return jsonify(result=str(model.predict(array)[0]))
    
    except ValueError:
        return jsonify(result=str(ValueError))
           
if __name__ == '__main__':
    app.run(port=5000,debug=True)
