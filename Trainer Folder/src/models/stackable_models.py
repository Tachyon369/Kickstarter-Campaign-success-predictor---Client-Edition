# D:\Trainer\src/models/stackable_models.py

"""
This file contains simple wrapper classes around the actual model implementations
(e.g., scikit-learn, XGBoost). These wrappers are designed specifically to be
instantiated with hyperparameters and then passed directly into a scikit-learn
StackingClassifier or StackingRegressor.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

# --- Stackable Wrappers ---

class StackableLogisticRegression:
    def __new__(cls, **kwargs):
        kwargs.setdefault('random_state', 42)
        return LogisticRegression(**kwargs)

class StackableRandomForest:
    def __new__(cls, task_type='classification', **kwargs):
        kwargs.setdefault('random_state', 42)
        kwargs.setdefault('n_jobs', -1)
        if task_type == 'classification':
            return RandomForestClassifier(**kwargs)
        else:
            return RandomForestRegressor(**kwargs)

class StackableSVM:
    def __new__(cls, task_type='classification', **kwargs):
        kwargs.setdefault('random_state', 42)
        if task_type == 'classification':
            return SVC(**kwargs)
        else:
            return SVR(**kwargs)

class StackableXGBoost:
    def __new__(cls, task_type='classification', device='cpu', **kwargs):
        kwargs.setdefault('random_state', 42)
        kwargs['tree_method'] = 'hist'
        if device == 'gpu':
            kwargs['device'] = 'cuda'
        
        if task_type == 'classification':
            return XGBClassifier(**kwargs)
        else:
            return XGBRegressor(**kwargs)

class StackableLightGBM:
    def __new__(cls, task_type='classification', device='cpu', **kwargs):
        kwargs.setdefault('random_state', 42)
        kwargs.setdefault('n_jobs', -1)
        if device == 'gpu':
            kwargs['device'] = 'gpu'
        
        if task_type == 'classification':
            return LGBMClassifier(**kwargs)
        else:
            return LGBMRegressor(**kwargs)

class StackableCatBoost:
    def __new__(cls, task_type='classification', device='cpu', **kwargs):
        kwargs.setdefault('random_state', 42)
        kwargs.setdefault('verbose', False)
        if device == 'gpu':
            kwargs['task_type'] = 'GPU'
            
        if task_type == 'classification':
            return CatBoostClassifier(**kwargs)
        else:
            return CatBoostRegressor(**kwargs)

# A dictionary to easily map the string names from the UI to these stackable classes
STACKABLE_MODELS_MAP = {
    "Logistic Regression": StackableLogisticRegression,
    "Random Forest": StackableRandomForest,
    "SVM (Support Vector Machine)": StackableSVM,
    "XGBoost": StackableXGBoost,
    "LightGBM": StackableLightGBM,
    "CatBoost": StackableCatBoost
    # We exclude the DL models from stacking for now due to their complex pipeline
}