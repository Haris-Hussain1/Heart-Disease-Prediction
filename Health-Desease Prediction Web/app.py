from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('model/DHC-1898 task 3 model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect inputs from the form
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    cp = int(request.form['cp'])
    trestbps = int(request.form['trestbps'])
    chol = int(request.form['chol'])
    fbs = int(request.form['fbs'])
    restecg = int(request.form['restecg'])
    thalach = int(request.form['thalach'])
    exang = int(request.form['exang'])
    oldpeak = float(request.form['oldpeak'])
    slope = int(request.form['slope'])
    ca = int(request.form['ca'])
    thal = int(request.form['thal'])
    
    # Arrange input in EXACT order
    features = np.array([age, sex, cp, trestbps, chol, fbs, restecg, 
                        thalach, exang, oldpeak, slope, ca, thal])
    
    # Reshape for prediction (1 sample, 13 features)
    features = features.reshape(1, -1)
    
    # Predict using model
    prediction = model.predict(features)
    
    # Output result
    if prediction[0] == 1:
        result = "High Risk of Heart Disease"
    else:
        result = "Low Risk of Heart Disease"
    
    return render_template('result.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
