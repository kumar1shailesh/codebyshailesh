class PredictionController:
    def __init__(self, feature_engineering_service, prediction_service):
        self.feature_engineering_service = feature_engineering_service
        self.prediction_service = prediction_service

    def predict(self, request_data):
        # Perform feature engineering on the request data
        processed_data = self.feature_engineering_service.process(request_data)
        
        # Make predictions using the pre-trained model
        predictions = self.prediction_service.predict(processed_data)
        
        return predictions
