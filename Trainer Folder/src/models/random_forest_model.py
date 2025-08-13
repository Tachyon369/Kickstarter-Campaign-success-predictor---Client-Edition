# D:\Trainer\src\models\random_forest_model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from .base_model import BaseModel

class RandomForestModel(BaseModel):
    """
    A wrapper for the scikit-learn Random Forest models that conforms to our architecture.
    """

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, device: str = 'cpu', **kwargs):
        """
        Instantiates and trains the Random Forest model.
        NOTE: This model is CPU-only but will use all available cores.
        """
        self._task_type = self._determine_task_type(y_train)
        
        # Start with the user-provided hyperparameters
        params = kwargs.copy()
        
        # Add non-tunable defaults
        # n_jobs=-1 tells scikit-learn to use all available CPU cores for training
        params.update({'random_state': 42, 'n_jobs': -1})

        if device == 'gpu':
            print("Warning: Random Forest (scikit-learn) does not support GPU. Training on CPU with all cores.")
        else:
            print("Training Random Forest on CPU with all cores...")

        # Instantiate the correct model with the updated params
        if self._task_type == 'classification':
            self.model = RandomForestClassifier(**params)
        else:
            self.model = RandomForestRegressor(**params)

        print(f"Training the Random Forest {self._task_type} model with params: {params}")
        self.model.fit(X_train, y_train)
        print("Training complete.")

    @property
    def task_type(self) -> str:
        return self._task_type

    def get_hyperparameters(self) -> dict:
        """
        Defines the default hyperparameters for the Random Forest model for the UI.
        """
        return {
            "n_estimators": 100,
            "max_depth": None, # Let the trees grow deep by default
            "min_samples_split": 2,
            "min_samples_leaf": 1
        }

    def get_info(self) -> dict:
        """Defines the model's name for the UI dropdown."""
        return {
            "name": "Random Forest",
            "type": "Classification / Regression"
        }

    def _determine_task_type(self, y: pd.Series) -> str:
        """Heuristic to determine if the task is classification or regression."""
        if y.nunique() < 25 and pd.api.types.is_integer_dtype(y.dtype):
            self._task_type = 'classification'
        else:
            self._task_type = 'regression'
        print(f"Random Forest task type detected: {self._task_type}")
        return self._task_type
        
    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """Makes predictions on new data."""
        if not hasattr(self, 'model'):
            raise RuntimeError("Model has not been trained yet. Please call train() first.")
        
        print(f"Making predictions with Random Forest {self._task_type} model...")
        predictions = self.model.predict(X_test)
        return pd.Series(predictions, index=X_test.index, name="Predicted")