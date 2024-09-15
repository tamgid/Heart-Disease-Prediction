from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the request is JSON
    if request.is_json:
        # Parse JSON data
        data = request.get_json()
        # Extract and convert features to integers
        int_features = [int(data[key]) for key in data.keys()]
        final_features = [np.array(int_features)]
        
        # Make prediction
        prediction = model.predict(final_features)
        output = 'The person has heart disease' if prediction[0] == 1 else 'The person does not have heart disease'

        # Return the prediction as a JSON response
        return jsonify({'prediction': output})
    else:
        return jsonify({'error': 'Request must be JSON'}), 400

if __name__ == "__main__":
    app.run(debug=True)
