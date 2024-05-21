import sys
import os
from flask import Flask, jsonify, request
from flask_cors import CORS

from xgboost_predict import get_prediction

# Add the 'project' directory to the Python path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_dir)

app = Flask(__name__)
CORS(app)

# CORS Headers
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

@app.route("/get_predictions", methods=['POST'])
def get_predictions():
    data = request.json

    # Define the expected input fields
    expected_fields = ['startDate', 'endDate']

    # Extract the input values in the correct order
    input_values = [data.get(field) for field in expected_fields]

    response = get_prediction(input_values[0], input_values[1])

    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)