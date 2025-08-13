# D:\Trainer\src\models\neural_network_model.py

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from .base_model import BaseModel

class NeuralNetworkModel(BaseModel):
    """
    A wrapper for the scikit-learn Multi-Layer Perceptron (MLP) models.
    """

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, device: str = 'cpu', **kwargs):
        """
        Instantiates and trains the MLP model.
        """
        self._task_type = self._determine_task_type(y_train)
        
        params = kwargs.copy()
        
        # --- Safe conversion for parameters that expect specific types ---
        try:
            # Convert hidden_layer_sizes string to a tuple
            if 'hidden_layer_sizes' in params:
                params['hidden_layer_sizes'] = eval(params['hidden_layer_sizes'])
            # Convert early_stopping string to a boolean
            if 'early_stopping' in params:
                params['early_stopping'] = (str(params['early_stopping']).lower() == 'true')
        except Exception as e:
            print(f"Warning: Could not parse a hyperparameter. Using defaults for it. Error: {e}")
            # In case of error, pop the problematic keys to avoid crashing the model
            params.pop('hidden_layer_sizes', None)
            params.pop('early_stopping', None)
            
        # Add non-tunable defaults
        params.update({'random_state': 42})

        if device == 'gpu':
            print("Warning: MLP (scikit-learn) does not support GPU. Training on CPU.")
        
        print("Training Neural Network (MLP) on CPU...")
        
        if self._task_type == 'classification':
            self.model = MLPClassifier(**params)
        else:
            self.model = MLPRegressor(**params)

        print(f"Training the MLP {self._task_type} model with params: {params}")
        self.model.fit(X_train, y_train)
        print("Training complete.")

    @property
    def task_type(self) -> str:
        return self._task_type

    def get_hyperparameters(self) -> dict:
        """
        Defines the default hyperparameters for the MLP model for the UI.
        """
        # <<< [THE CHANGE] Added max_iter and early_stopping >>>
        return {
            "hidden_layer_sizes": "(100,)", 
            "activation": "relu",          
            "solver": "adam",              
            "alpha": 0.0001,               
            "learning_rate": "constant",   
            "max_iter": 500,               # Max number of epochs (training steps)
            "early_stopping": "True"       # Use "True" or "False" as strings
        }

    def get_info(self) -> dict:
        """Defines the model's name for the UI dropdown."""
        return {
            "name": "Neural Network (MLP)",
            "type": "Classification / Regression"
        }

    def _determine_task_type(self, y: pd.Series) -> str:
        # ... (This method is unchanged) ...
        if y.nunique() < 25 and pd.api.types.is_integer_dtype(y.dtype):
            self._task_type = 'classification'
        else:
            self._task_type = 'regression'
        print(f"MLP task type detected: {self._task_type}")
        return self._task_type
        
    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        # ... (This method is unchanged) ...
        if not hasattr(self, 'model'):
            raise RuntimeError("Model has not been trained yet. Please call train() first.")
        
        print(f"Making predictions with MLP {self._task_type} model...")
        predictions = self.model.predict(X_test)
        return pd.Series(predictions, index=X_test.index, name="Predicted")