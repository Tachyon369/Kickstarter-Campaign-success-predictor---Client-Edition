# D:\Trainer\src\models\logistic_regression.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
from .base_model import BaseModel
import numpy as np

class LogisticRegressionModel(BaseModel):
    """
    A wrapper for the scikit-learn Logistic Regression model that conforms
    to our final architecture. It has no __init__ method and creates the model
    inside the train() method.
    """
    
    # NOTE: No __init__ method here. This is crucial for the discovery process.

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, device: str = 'cpu', **kwargs):
        """
        Instantiates and trains the Logistic Regression model using hyperparameters from the UI.
        """
        if device == 'gpu':
            print("Warning: Logistic Regression does not support GPU. Training on CPU instead.")
        
        # The model is created here, inside train(), using the kwargs
        self.model = LogisticRegression(random_state=42, **kwargs)
        
        print(f"Training LogisticRegressionModel with params: {kwargs}")
        self.model.fit(X_train, y_train)

    @property
    def task_type(self) -> str:
        """This model is only for classification."""
        return 'classification'
    
    def get_hyperparameters(self) -> dict:
        """Defines the default hyperparameters for the UI."""
        return {
            "C": 1.0,
            "solver": "lbfgs"
        }

    def get_info(self) -> dict:
        """Defines the model's name for the UI dropdown."""
        return {
            "name": "Logistic Regression",
            "type": "Classification"
        }
        
    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """Makes predictions on new data."""
        if not hasattr(self, 'model'):
            raise RuntimeError("Model has not been trained yet. Please call train() first.")
        return self.model.predict(X_test)


    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Makes probability predictions on new data.
        This method is required for the predictor app to show confidence scores.
        """
        if not hasattr(self, 'model'):
            raise RuntimeError("Model has not been trained yet. Please call train() first.")
        # This calls the actual predict_proba method of the scikit-learn model
        return self.model.predict_proba(X_test)