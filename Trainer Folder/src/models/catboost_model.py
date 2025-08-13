# D:\Trainer\src\models\catboost_model.py

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from .base_model import BaseModel

class CatBoostModel(BaseModel):
    """
    A wrapper for the CatBoost library that conforms to our application's architecture.
    """

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, device: str = 'cpu', **kwargs):
        """
        Instantiates and trains the CatBoost model.
        CatBoost uses task_type='GPU' for hardware acceleration.
        """
        self._task_type = self._determine_task_type(y_train)
        
        # Start with the user-provided hyperparameters
        params = kwargs.copy()
        
        # Add non-tunable defaults
        # verbose=False prevents CatBoost from printing excessive logs during training
        params.update({'random_state': 42, 'verbose': False})

        if device == 'gpu':
            print("Attempting to train CatBoost on GPU...")
            # For CatBoost, GPU support is enabled via the task_type parameter
            params['task_type'] = 'GPU'
        else:
            print("Training CatBoost on CPU...")
            # No change needed, CPU is the default task_type

        # Instantiate the correct model with the updated params
        if self._task_type == 'classification':
            self.model = CatBoostClassifier(**params)
        else:
            self.model = CatBoostRegressor(**params)

        print(f"Training the CatBoost {self._task_type} model with params: {params}")
        self.model.fit(X_train, y_train)
        print("Training complete.")

    @property
    def task_type(self) -> str:
        return self._task_type

    def get_hyperparameters(self) -> dict:
        """
        Defines the default hyperparameters for the CatBoost model for the UI.
        """
        return {
            "iterations": 200,      # Equivalent to n_estimators
            "learning_rate": 0.1,
            "depth": 6,             # Equivalent to max_depth
            "l2_leaf_reg": 3.0      # L2 regularization term
        }

    def get_info(self) -> dict:
        """Defines the model's name for the UI dropdown."""
        return {
            "name": "CatBoost",
            "type": "Classification / Regression"
        }

    def _determine_task_type(self, y: pd.Series) -> str:
        """Heuristic to determine if the task is classification or regression."""
        if y.nunique() < 25 and pd.api.types.is_integer_dtype(y.dtype):
            self._task_type = 'classification'
        else:
            self._task_type = 'regression'
        print(f"CatBoost task type detected: {self._task_type}")
        return self._task_type
        
    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """Makes predictions on new data."""
        if not hasattr(self, 'model'):
            raise RuntimeError("Model has not been trained yet. Please call train() first.")
        
        print(f"Making predictions with CatBoost {self._task_type} model...")
        predictions = self.model.predict(X_test)
        return pd.Series(predictions, index=X_test.index, name="Predicted")