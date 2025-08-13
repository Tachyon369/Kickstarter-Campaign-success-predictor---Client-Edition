# D:\Trainer\src\models\linear_regression_model.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from .base_model import BaseModel

class LinearRegressionModel(BaseModel):
    """
    A wrapper for the scikit-learn Linear Regression model.
    This serves as a simple, fast baseline for regression tasks.
    """

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, device: str = 'cpu', **kwargs):
        """
        Instantiates and trains the Linear Regression model.
        This model does not have tunable hyperparameters from our UI's perspective.
        """
        self._task_type = 'regression' # This model is only for regression

        if device == 'gpu':
            print("Warning: Linear Regression (scikit-learn) does not support GPU. Training on CPU.")
        else:
            print("Training Linear Regression on CPU...")

        # No hyperparameters from our UI are used, but we accept **kwargs for compatibility
        self.model = LinearRegression()

        print(f"Training the Linear Regression model...")
        self.model.fit(X_train, y_train)
        print("Training complete.")

    @property
    def task_type(self) -> str:
        """This model is only for regression tasks."""
        return 'regression'

    def get_hyperparameters(self) -> dict:
        """
        Linear Regression (in its basic form) has no hyperparameters for us to tune.
        We return an empty dictionary.
        """
        return {}

    def get_info(self) -> dict:
        """Defines the model's name for the UI dropdown."""
        return {
            "name": "Simple Linear Regression",
            "type": "Regression"
        }
        
    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """Makes predictions on new data."""
        if not hasattr(self, 'model'):
            raise RuntimeError("Model has not been trained yet. Please call train() first.")
        
        print(f"Making predictions with Linear Regression model...")
        predictions = self.model.predict(X_test)
        return pd.Series(predictions, index=X_test.index, name="Predicted")