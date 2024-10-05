from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained models when the app starts
rfr = joblib.load('random_forest_model.pkl')
dtr = joblib.load('decision_tree_model.pkl')
xgb_model = joblib.load('xgboost_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_ppv():
    data = request.form  # Get form data
    input_features = np.array([float(data['feature1']), float(data['feature2']), float(data['feature3']), 
                               float(data['feature4']), float(data['feature5']), float(data['feature6']), 
                               float(data['feature7']), float(data['feature8'])])

    # Predict with Random Forest (you can change to other models if needed)
    prediction = xgb_model.predict([input_features])  # Using the loaded model
    
    return jsonify({'predicted_ppv': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
