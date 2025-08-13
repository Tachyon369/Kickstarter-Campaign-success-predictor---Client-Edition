# D:\Trainer\src\models\svm_model.py

import pandas as pd
import numpy as np
from sklearn.svm import SVC, SVR
from .base_model import BaseModel

class SVMModel(BaseModel):
    """
    A wrapper for the scikit-learn Support Vector Machine models (SVC and SVR)
    that conforms to our application's architecture.
    """

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, device: str = 'cpu', **kwargs):
        """
        Instantiates and trains the SVM model.
        NOTE: This model is CPU-only. The 'device' parameter is ignored.
        """
        self._task_type = self._determine_task_type(y_train)
        
        # Start with the user-provided hyperparameters
        params = kwargs.copy()
        
        # Add non-tunable defaults
        # probability=True is needed for some advanced functions, but can slow down training.
        # We can add it later if needed. For now, we omit it for speed.
        params.update({'random_state': 42})

        if device == 'gpu':
            print("Warning: SVM (scikit-learn) does not support GPU. Training on CPU.")
        else:
            print("Training SVM on CPU...")

        # Instantiate the correct model with the updated params
        if self._task_type == 'classification':
            # Support Vector Classifier
            self.model = SVC(**params)
        else:
            # Support Vector Regressor
            self.model = SVR(**params)

        print(f"Training the SVM {self._task_type} model with params: {params}")
        self.model.fit(X_train, y_train)
        print("Training complete.")

    @property
    def task_type(self) -> str:
        return self._task_type

    def get_hyperparameters(self) -> dict:
        """
        Defines the default hyperparameters for the SVM model for the UI.
        """
        return {
            "C": 1.0,               # Regularization parameter
            "kernel": "rbf",        # Kernel type: 'linear', 'poly', 'rbf', 'sigmoid'
            "gamma": "scale"        # Kernel coefficient
        }

    def get_info(self) -> dict:
        """Defines the model's name for the UI dropdown."""
        return {
            "name": "SVM (Support Vector Machine)",
            "type": "Classification / Regression"
        }

    def _determine_task_type(self, y: pd.Series) -> str:
        """Heuristic to determine if the task is classification or regression."""
        if y.nunique() < 25 and pd.api.types.is_integer_dtype(y.dtype):
            self._task_type = 'classification'
        else:
            self._task_type = 'regression'
        print(f"SVM task type detected: {self._task_type}")
        return self._task_type
        
    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """Makes predictions on new data."""
        if not hasattr(self, 'model'):
            raise RuntimeError("Model has not been trained yet. Please call train() first.")
        
        print(f"Making predictions with SVM {self._task_type} model...")
        predictions = self.model.predict(X_test)
        return pd.Series(predictions, index=X_test.index, name="Predicted")