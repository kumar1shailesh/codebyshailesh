from sklearn.linear_model import LinearRegression
import pickle

class ModelTrainingService:
    def train(self, X, y):
        # Implement your model training logic here
        model = LinearRegression()
        model.fit(X, y)
        
        # Save the trained model to a file
        with open('app/models/pre_trained_model.pkl', 'wb') as f:
            pickle.dump(model, f)
