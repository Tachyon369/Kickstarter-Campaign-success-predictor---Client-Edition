# D:\Trainer\src\models\base_model.py
from abc import ABC, abstractmethod
import pandas as pd

class BaseModel(ABC):
    @property
    @abstractmethod
    def task_type(self) -> str: pass
    
    @abstractmethod
    def get_hyperparameters(self) -> dict:
        """ Returns a dictionary defining the model's tunable hyperparameters. """
        pass

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, device: str = 'cpu', **kwargs):
        pass
    
    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> pd.Series: pass

    @abstractmethod
    def get_info(self) -> dict: pass