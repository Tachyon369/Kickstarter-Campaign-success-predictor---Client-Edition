 # D:\Trainer\src\models\xgboost_model.py
import pandas as pd
import numpy as np
import xgboost as xgb
from .base_model import BaseModel

class XGBoostModel(BaseModel):
    def __init__(self):
        self.model = None
        self._task_type = 'unknown'
        self._device = 'cpu'
        print(f"XGBoostModel instance created.")

    def get_hyperparameters(self) -> dict:
        return {
            "n_estimators": 100,
            "learning_rate": 0.3,
            "max_depth": 6
        }

    @property
    def task_type(self) -> str: return self._task_type
    def get_info(self) -> dict: return {"name": "XGBoost", "type": "Classification / Regression"}
    def _determine_task_type(self, y: pd.Series) -> str:
        if y.nunique() < 25 and pd.api.types.is_integer_dtype(y.dtype): return 'classification'
        else: return 'regression'

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, device: str = 'cpu', **kwargs):
        self._task_type = self._determine_task_type(y_train)
        self._device = device
        params = kwargs.copy()
        params.update({'random_state': 42, 'tree_method': 'hist'})
        if device == 'gpu':
            try:
                xgb.XGBClassifier(device='cuda')
                params['device'] = 'cuda'
            except xgb.core.XGBoostError: self._device = 'cpu'
        if self._task_type == 'classification': self.model = xgb.XGBClassifier(**params)
        else: self.model = xgb.XGBRegressor(**params)
        print(f"Training the {self._task_type} model with params: {params}")
        self.model.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        if self.model is None: raise RuntimeError("Model has not been trained yet.")
        if self._device == 'gpu':
            try: self.model.get_booster().set_param('device', 'cuda')
            except Exception as e: print(f"Warning: Could not set prediction device to CUDA. Error: {e}")
        return self.model.predict(X_test)