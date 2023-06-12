from flask import Blueprint, request, jsonify
from app.api.controllers.prediction_controller import PredictionController

prediction_api = Blueprint('prediction_api', __name__)
prediction_controller = PredictionController()

@prediction_api.route('/predict', methods=['POST'])
def predict():
    request_data = request.json
    
    predictions = prediction_controller.predict(request_data)
    
    return jsonify({'predictions': predictions})
