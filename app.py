from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('XGBoost_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        Pclass = int(request.form['Pclass'])
        Sex = request.form['Sex']
        Age = float(request.form['Age'])
        Fare = float(request.form['Fare'])
        Title_Mr = int(request.form['Title_Mr'])
        
        # Convert Sex to binary
        if Sex == 'male':
            Sex = 1
        else:
            Sex = 0
        
        # Features array
        features = np.array([[Pclass, Sex, Age, Fare, Title_Mr]])
        
        # Make prediction
        prediction = model.predict(features)
        
        # Result interpretation
        if prediction[0] == 1:
            result = "Survived"
        else:
            result = "Did not survive"
        
        return render_template('index.html', prediction_text=f'Prediction: {result}')

if __name__ == "__main__":
    app.run(debug=True)