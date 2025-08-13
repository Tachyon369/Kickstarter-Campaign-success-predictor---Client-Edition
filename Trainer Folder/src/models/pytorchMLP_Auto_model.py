# D:\Trainer\src\models\pytorch_model.py

# ... (all imports and helper classes are unchanged) ...
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
class TabularDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        if y is not None: self.y = torch.tensor(y.values, dtype=torch.float32).squeeze()
        else: self.y = None
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        if self.y is not None: return self.X[idx], self.y[idx]
        return self.X[idx]
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, task_type):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, output_dim)
        )
        self.task_type = task_type
    def forward(self, x):
        x = self.layers(x)
        if self.task_type == 'classification': return x.squeeze()
        return x

class PyTorchModel(BaseModel):
    # ... (__init__, task_type, get_info, get_hyperparameters, _determine_task_type are unchanged) ...
    def __init__(self):
        if not PYTORCH_AVAILABLE: raise ImportError("PyTorch not installed.")
        self.model = None
        self._task_type = 'unknown'
        self.device = 'cpu'
        self.output_dim = 1
    @property
    def task_type(self) -> str: return self._task_type
    def get_info(self) -> dict: return {"name": "PyTorch MLP", "type": "Classification / Regression"}
    def get_hyperparameters(self) -> dict: return {"learning_rate": 0.001, "epochs": 10, "batch_size": 32}
    def _determine_task_type(self, y: pd.Series) -> str:
        if y.nunique() < 25 and pd.api.types.is_integer_dtype(y.dtype):
            self.output_dim = y.nunique()
            if self.output_dim == 2: self.output_dim = 1
            return 'classification'
        else:
            self.output_dim = 1
            return 'regression'

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, device: str = 'cpu', **kwargs):
        self._task_type = self._determine_task_type(y_train)
        
        # <<< [THE FIX] Translate the generic 'gpu' string to the specific 'cuda' string >>>
        pytorch_device_str = "cuda" if device == "gpu" else "cpu"
        self.device = torch.device(pytorch_device_str if torch.cuda.is_available() else "cpu")
        # The rest of the function is unchanged
        
        print(f"PyTorch will train on device: {self.device}")
        
        train_dataset = TabularDataset(X_train, y_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=kwargs.get('batch_size', 32), shuffle=True)
        
        input_dim = X_train.shape[1]
        self.model = MLP(input_dim=input_dim, output_dim=self.output_dim, task_type=self._task_type).to(self.device)
        
        if self._task_type == 'classification':
            criterion = nn.BCEWithLogitsLoss() if self.output_dim == 1 else nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=kwargs.get('learning_rate', 0.001))
        
        print("Starting PyTorch training loop...")
        self.model.train()
        for epoch in range(kwargs.get('epochs', 10)):
            epoch_loss = 0
            for i, (features, labels) in enumerate(train_loader):
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            print(f'Epoch [{epoch+1}/{kwargs.get("epochs", 10)}], Loss: {epoch_loss/len(train_loader):.4f}')
        print("Training complete.")

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        # ... (This method is unchanged) ...
        if not self.model: raise RuntimeError("Model not trained.")
        print("Making predictions with PyTorch model...")
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
                else:
                    preds = outputs
                all_preds.extend(preds.cpu().numpy())
        return pd.Series(np.array(all_preds).flatten(), index=X_test.index, name="Predicted")