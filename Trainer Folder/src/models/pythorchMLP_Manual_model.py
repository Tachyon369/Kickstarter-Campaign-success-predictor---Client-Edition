# D:\Trainer\src\models\pytorch_model.py

import pandas as pd
import numpy as np
from .base_model import BaseModel

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    class nn: Module = object
    class Dataset: pass

# --- TabularDataset Helper Class (Unchanged) ---
class TabularDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        if y is not None: self.y = torch.tensor(y.values, dtype=torch.float32).squeeze()
        else: self.y = None
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        if self.y is not None: return self.X[idx], self.y[idx]
        return self.X[idx]

# --- [UPGRADED] The Dynamic Neural Network Architecture ---
class DynamicMLP(nn.Module):
    def __init__(self, input_dim, output_dim, task_type,
                 hidden_layers=2, neurons_per_layer=128, dropout_rate=0.3):
        super(DynamicMLP, self).__init__()
        
        layers = []
        current_dim = input_dim

        # Dynamically create hidden layers in a loop
        for i in range(hidden_layers):
            layers.append(nn.Linear(current_dim, neurons_per_layer))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = neurons_per_layer # The input for the next layer is the output of this one

        # Add the final output layer
        layers.append(nn.Linear(current_dim, output_dim))
        
        # Create the sequential model from the list of layers
        self.layers = nn.Sequential(*layers)
        self.task_type = task_type
        print(f"Created DynamicMLP with structure:\n{self.layers}")

    def forward(self, x):
        x = self.layers(x)
        if self.task_type == 'classification': return x.squeeze()
        return x

# --- Main Model Wrapper (Modified) ---
class PyTorchModel(BaseModel):
    def __init__(self):
        if not PYTORCH_AVAILABLE: raise ImportError("PyTorch not installed.")
        self.model = None
        self._task_type = 'unknown'
        self.device = 'cpu'
        self.output_dim = 1

    @property
    def task_type(self) -> str: return self._task_type
    def get_info(self) -> dict: return {"name": "PyTorch Manual MLP", "type": "Classification / Regression"}

    # <<< [MODIFIED] Added architecture hyperparameters >>>
    def get_hyperparameters(self) -> dict:
        return {
            "learning_rate": 0.001,
            "epochs": 10,
            "batch_size": 32,
            "hidden_layers": 2,      # Number of hidden layers
            "neurons_per_layer": 128,  # Neurons in each hidden layer
            "dropout_rate": 0.3      # Regularization to prevent overfitting
        }

    def _determine_task_type(self, y: pd.Series) -> str:
        # (This method is unchanged)
        if y.nunique() < 25 and pd.api.types.is_integer_dtype(y.dtype):
            self.output_dim = y.nunique()
            if self.output_dim == 2: self.output_dim = 1
            return 'classification'
        else:
            self.output_dim = 1
            return 'regression'

    # <<< [MODIFIED] train() now extracts architecture params and passes them to the MLP >>>
    def train_old(self, X_train: pd.DataFrame, y_train: pd.Series, device: str = 'cpu', **kwargs):
        self._task_type = self._determine_task_type(y_train)
        pytorch_device_str = "cuda" if device == "gpu" else "cpu"
        self.device = torch.device(pytorch_device_str if torch.cuda.is_available() else "cpu")
        print(f"PyTorch will train on device: {self.device}")

        # Extract architecture and training parameters
        arch_params = {
            'hidden_layers': kwargs.get('hidden_layers', 2),
            'neurons_per_layer': kwargs.get('neurons_per_layer', 128),
            'dropout_rate': kwargs.get('dropout_rate', 0.3)
        }
        train_params = {
            'batch_size': kwargs.get('batch_size', 32),
            'learning_rate': kwargs.get('learning_rate', 0.001),
            'epochs': kwargs.get('epochs', 10)
        }

        train_dataset = TabularDataset(X_train, y_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=train_params['batch_size'], shuffle=True)
        
        input_dim = X_train.shape[1]
        # Pass the architecture parameters to the DynamicMLP
        self.model = DynamicMLP(
            input_dim=input_dim, output_dim=self.output_dim, task_type=self._task_type, **arch_params
        ).to(self.device)
        
        if self._task_type == 'classification':
            criterion = nn.BCEWithLogitsLoss() if self.output_dim == 1 else nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=train_params['learning_rate'])
        
        print("Starting PyTorch training loop...")
        self.model.train()
        for epoch in range(train_params['epochs']):
            epoch_loss = 0
            for features, labels in train_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            print(f'Epoch [{epoch+1}/{train_params["epochs"]}], Loss: {epoch_loss/len(train_loader):.4f}')
        print("Training complete.")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, device: str = 'cpu', **kwargs):
        self._task_type = self._determine_task_type(y_train)
        pytorch_device_str = "cuda" if device == "gpu" else "cpu"
        self.device = torch.device(pytorch_device_str if torch.cuda.is_available() else "cpu")
        print(f"PyTorch will train on device: {self.device}")

        arch_params = {
            'hidden_layers': kwargs.get('hidden_layers', 2),
            'neurons_per_layer': kwargs.get('neurons_per_layer', 128),
            'dropout_rate': kwargs.get('dropout_rate', 0.3)
        }
        train_params = {
            'batch_size': kwargs.get('batch_size', 32),
            'learning_rate': kwargs.get('learning_rate', 0.001),
            'epochs': kwargs.get('epochs', 10)
        }

        train_dataset = TabularDataset(X_train, y_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=train_params['batch_size'], shuffle=True)

        input_dim = X_train.shape[1]
        self.model = DynamicMLP(
            input_dim=input_dim, output_dim=self.output_dim, task_type=self._task_type, **arch_params
        ).to(self.device)

        if self._task_type == 'classification':
            criterion = nn.BCEWithLogitsLoss() if self.output_dim == 1 else nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=train_params['learning_rate'])

        print("Starting PyTorch training loop...")
        self.model.train()

        for epoch in range(train_params['epochs']):
            epoch_loss = 0
            for features, labels in train_loader:
                features, labels = features.to(self.device), labels.to(self.device)

                outputs = self.model(features)

                if isinstance(criterion, nn.CrossEntropyLoss):
                    loss = criterion(outputs, labels.long())
                else:  # For BCEWithLogitsLoss or MSELoss
                    loss = criterion(outputs.squeeze(-1), labels.float())

                # Check for nan loss before proceeding
                if torch.isnan(loss):
                    print("NaN loss detected. Stopping training.")
                    print("Try lowering the learning rate, scaling your data, or checking for bad data.")
                    return  # Exit the function

                optimizer.zero_grad()
                loss.backward()

                # --- START OF GRADIENT CLIPPING ---
                # Add this line to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                # --- END OF GRADIENT CLIPPING ---

                optimizer.step()

                epoch_loss += loss.item()

            print(f'Epoch [{epoch + 1}/{train_params["epochs"]}], Loss: {epoch_loss / len(train_loader):.4f}')

        print("Training complete.")

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        # (This method is unchanged)
        if not self.model: raise RuntimeError("Model not trained.")
        self.model.eval()
        all_preds = []
        test_dataset = TabularDataset(X_test)
        test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
        with torch.no_grad():
            for features in test_loader:
                features = features.to(self.device)
                outputs = self.model(features)
                if self._task_type == 'classification':
                    if self.output_dim == 1: preds = torch.sigmoid(outputs) > 0.5
                    else: _, preds = torch.max(outputs, 1)
                else: preds = outputs
                all_preds.extend(preds.cpu().numpy())
        return pd.Series(np.array(all_preds).flatten(), index=X_test.index, name="Predicted")