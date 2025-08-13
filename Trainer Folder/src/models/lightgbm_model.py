# D:\Trainer\src\models\lightgbm_model.py

import pandas as pd
import numpy as np
import lightgbm as lgb
from .base_model import BaseModel

class LightGBMModel(BaseModel):
    """
    A wrapper for the LightGBM library that conforms to our application's architecture.
    """

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, device: str = 'cpu', **kwargs):
        """
        Instantiates and trains the LightGBM model.
        """
        self._task_type = self._determine_task_type(y_train)
        
        # Start with the user-provided hyperparameters
        params = kwargs.copy()
        
        # Add non-tunable defaults
        params.update({'random_state': 42, 'n_jobs': -1})

        if device == 'gpu':
            print("Attempting to train LightGBM on GPU...")
            params['device'] = 'gpu'
        else:
            print("Training LightGBM on CPU...")
            params['device'] = 'cpu'

        # Instantiate the correct model with the updated params
        if self._task_type == 'classification':
            self.model = lgb.LGBMClassifier(**params)
        else:
            self.model = lgb.LGBMRegressor(**params)

        print(f"Training the LightGBM {self._task_type} model with params: {params}")
        self.model.fit(X_train, y_train)
        print("Training complete.")

    @property
    def task_type(self) -> str:
        return self._task_type

    def get_hyperparameters(self) -> dict:
        """
        Defines the default hyperparameters for the LightGBM model for the UI.
        """
        return {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "max_depth": -1
        }

    def get_info(self) -> dict:
        """Defines the model's name for the UI dropdown."""
        return {
            "name": "LightGBM",
            "type": "Classification / Regression"
        }

    def _determine_task_type(self, y: pd.Series) -> str:
        """Heuristic to determine if the task is classification or regression."""
        if y.nunique() < 25 and pd.api.types.is_integer_dtype(y.dtype):
            self._task_type = 'classification'
        else:
            self._task_type = 'regression'
        print(f"LightGBM task type detected: {self._task_type}")
        return self._task_type
        
    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """Makes predictions on new data."""
        if not hasattr(self, 'model'):
            raise RuntimeError("Model has not been trained yet. Please call train() first.")
        
        print(f"Making predictions with LightGBM {self._task_type} model...")
        predictions = self.model.predict(X_test)
        return pd.Series(predictions, index=X_test.index, name="Predicted")

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Makes probability predictions on new data.
        This is only valid for classification tasks.
        """
        if not hasattr(self, 'model'):
            raise RuntimeError("Model has not been trained yet. Please call train() first.")
        
        if self.task_type != 'classification':
            raise NotImplementedError("predict_proba is only available for classification tasks.")
        
        # This calls the actual predict_proba method of the scikit-learn compatible model
        return self.model.predict_proba(X_test)