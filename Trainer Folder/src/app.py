# D:\Trainer\src\app.py

import gradio as gr
import pandas as pd
import numpy as np
import os
import importlib
import inspect
import sys
import subprocess
from datetime import datetime
import joblib
from models.base_model import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import json
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, LabelEncoder

# --- Configuration & Function Definitions ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODELS_DIR = os.path.join(ROOT_DIR, 'src', 'models')
MODELS_OUTPUT_DIR = os.path.join(ROOT_DIR, 'models_output')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')

def discover_models():
    models = {}
    for filename in os.listdir(MODELS_DIR):
        if filename.endswith(".py") and not filename.startswith(("__", "base_")):
            module_name = f"models.{filename[:-3]}"
            try:
                module = importlib.import_module(module_name)
                importlib.reload(module)
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, BaseModel) and obj is not BaseModel:
                        instance = obj()
                        info = instance.get_info()
                        model_display_name = info.get("name", name)
                        models[model_display_name] = obj
            except ImportError as e: print(f"Warning: Could not import {module_name}. Error: {e}")
    return models

def discover_datasets(): return [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
def discover_saved_models(): return [f for f in os.listdir(MODELS_OUTPUT_DIR) if f.endswith("_model.joblib")]
def open_folder(path):
    if sys.platform == "win32": os.startfile(path)
    elif sys.platform == "darwin": subprocess.Popen(["open", path])
    else: subprocess.Popen(["xdg-open", path])

def refresh_all_dropdowns():
    datasets = discover_datasets()
    models = list(discover_models().keys())
    return gr.update(choices=datasets), gr.update(choices=models)

def load_target_columns(dataset_name):
    if not dataset_name: return gr.update(interactive=False)
    try:
        data_path = os.path.join(DATA_DIR, dataset_name)
        df_sample = pd.read_csv(data_path, nrows=1)
        return gr.update(choices=df_sample.columns.tolist(), value=None, interactive=True)
    except Exception as e: return gr.update(interactive=False)

def identify_features(dataset_name, target_col):
    if not target_col or not dataset_name: return None, None, None
    try:
        data_path = os.path.join(DATA_DIR, dataset_name)
        df_sample = pd.read_csv(data_path, nrows=50)
        feature_cols = [col for col in df_sample.columns if col != target_col]
        df_features = df_sample[feature_cols]
        numerical_cols = df_features.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df_features.select_dtypes(include=['object', 'category']).columns.tolist()
        return feature_cols, numerical_cols, categorical_cols
    except Exception: return None, None, None

def update_numerical_dropdown(dataset_name, target_col):
    feature_cols, numerical_cols, _ = identify_features(dataset_name, target_col)
    if feature_cols is None: return gr.update(interactive=False)
    return gr.update(choices=feature_cols, value=numerical_cols, interactive=True)

def update_onehot_dropdown(dataset_name, target_col):
    feature_cols, _, categorical_cols = identify_features(dataset_name, target_col)
    if feature_cols is None: return gr.update(interactive=False)
    return gr.update(choices=feature_cols, value=categorical_cols, interactive=True)

def update_label_dropdown(dataset_name, target_col):
    feature_cols, _, _ = identify_features(dataset_name, target_col)
    if feature_cols is None: return gr.update(interactive=False)
    return gr.update(choices=feature_cols, value=[], interactive=True)

def toggle_test_data_dropdown(split_method):
    return gr.update(visible=(split_method == "Select Test data csv"))

def update_hyperparameters_ui(model_name):
    if not model_name: return gr.update(value="", visible=False)
    try:
        model_class = discover_models()[model_name]
        default_params = model_class().get_hyperparameters()
        params_str = json.dumps(default_params, indent=4)
        return gr.update(value=params_str, visible=True)
    except Exception as e:
        print(f"Error loading hyperparameters: {e}")
        return gr.update(value="", visible=False)

def train_and_evaluate(model_name, dataset_name, compute_device, target_col, 
                       encode_target, # <-- NEW INPUT
                       numerical_cols, onehot_cols, label_cols,
                       scale_numericals, hyperparams_str,
                       split_method, test_dataset_name):
    try:
        hyperparams = json.loads(hyperparams_str)
        feature_cols = numerical_cols + onehot_cols + label_cols
        if not target_col or not feature_cols:
            gr.Error("Target and at least one feature column must be selected!")
            return None, None, "Error: Missing column selections."
        
        train_df = pd.read_csv(os.path.join(DATA_DIR, dataset_name))
        test_df = None
        if split_method == "Select Test data csv" and test_dataset_name:
            test_df = pd.read_csv(os.path.join(DATA_DIR, test_dataset_name))

        if split_method == "Auto-split (80/20)":
            X_train, X_test, y_train, y_test = train_test_split(train_df[feature_cols], train_df[target_col], test_size=0.2, random_state=42)
        else:
            X_train, y_train = train_df[feature_cols], train_df[target_col]
            if test_df is not None:
                X_test, y_test = test_df[feature_cols], test_df[target_col]
            else:
                X_test, y_test = X_train.copy(), y_train.copy()
        
        target_encoder = None
        if encode_target:
            target_encoder = LabelEncoder()
            # Fit on the combined data to learn all possible classes
            combined_y = pd.concat([y_train, y_test], axis=0, ignore_index=True)
            target_encoder.fit(combined_y)
            y_train = pd.Series(target_encoder.transform(y_train), index=y_train.index)
            y_test = pd.Series(target_encoder.transform(y_test), index=y_test.index)
            print(f"Target column '{target_col}' encoded. Classes: {target_encoder.classes_}")

        numerical_transformer = StandardScaler() if scale_numericals else 'passthrough'
        preprocessor = ColumnTransformer(transformers=[("numeric", numerical_transformer, numerical_cols), ("onehot", OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_cols), ("label", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), label_cols)], remainder='passthrough')
        preprocessor.set_output(transform="pandas")
        X_train_processed = preprocessor.fit_transform(X_train).astype(np.float32)
        X_test_processed = preprocessor.transform(X_test).astype(np.float32)
        
        model = discover_models()[model_name]()
        model.train(X_train_processed, y_train, device=compute_device.lower(), **hyperparams)
        
        predictions = model.predict(X_test_processed)
        metrics_dict = {}
        if model.task_type == 'classification':
            metrics_dict = {"Accuracy": round(accuracy_score(y_test, predictions), 4), "F1-Score": round(f1_score(y_test, predictions, average='weighted', zero_division=0), 4)}
        elif model.task_type == 'regression':
            metrics_dict = {"R-squared": round(r2_score(y_test, predictions), 4), "MSE": round(mean_squared_error(y_test, predictions), 4)}
        
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        base_name = f"{dataset_name.replace('.csv', '')}__{model_name.replace(' ', '_')}__{timestamp}"
        joblib.dump(preprocessor, os.path.join(MODELS_OUTPUT_DIR, f"{base_name}_preprocessor.joblib"))
        joblib.dump(model, os.path.join(MODELS_OUTPUT_DIR, f"{base_name}_model.joblib"))
        if target_encoder:
            joblib.dump(target_encoder, os.path.join(MODELS_OUTPUT_DIR, f"{base_name}_target_encoder.joblib"))
            print("Saved fitted target encoder.")
        
        with open(os.path.join(OUTPUT_DIR, f"{base_name}_metrics.txt"), 'w') as f: f.write(json.dumps(metrics_dict, indent=4))
        pd.DataFrame({'Actual': y_test.values, 'Predicted': predictions}).to_csv(os.path.join(OUTPUT_DIR, f"{base_name}_predictions.csv"), index=False)
        
        return pd.DataFrame(X_test_processed.head()), metrics_dict, f"Success! Artifacts saved with base name:\n{base_name}"
    except Exception as e:
        import traceback
        error_message = f"An error occurred: {e}"
        traceback.print_exc()
        return None, {"Error": str(e)}, error_message

def run_inference(saved_model_filename, test_dataset_name, test_target_col):
    try:
        model_path = os.path.join(MODELS_OUTPUT_DIR, saved_model_filename)
        preprocessor_path = model_path.replace("_model.joblib", "_preprocessor.joblib")
        target_encoder_path = model_path.replace("_model.joblib", "_target_encoder.joblib")
        
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        target_encoder = joblib.load(target_encoder_path) if os.path.exists(target_encoder_path) else None

        test_df = pd.read_csv(os.path.join(DATA_DIR, test_dataset_name))
        X_test_processed = preprocessor.transform(test_df)
        predictions = model.predict(X_test_processed)
        
        predictions_df = test_df.copy()
        
        metrics_dict = {"Info": "No target column provided."}
        if test_target_col and test_target_col in test_df.columns:
            y_test_original = test_df[test_target_col]
            y_test_for_metrics = y_test_original
            if target_encoder:
                y_test_for_metrics = target_encoder.transform(y_test_original)
                predictions_df['Predicted'] = target_encoder.inverse_transform(predictions)
            else:
                predictions_df['Predicted'] = predictions
            
            if model.task_type == 'classification':
                metrics_dict = {"Accuracy": round(accuracy_score(y_test_for_metrics, predictions), 4), "F1-Score": round(f1_score(y_test_for_metrics, predictions, average='weighted', zero_division=0), 4)}
            elif model.task_type == 'regression':
                metrics_dict = {"R-squared": round(r2_score(y_test_for_metrics, predictions), 4), "MSE": round(mean_squared_error(y_test_for_metrics, predictions), 4)}
        else:
            predictions_df['Predicted'] = target_encoder.inverse_transform(predictions) if target_encoder else predictions

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        model_base_name = saved_model_filename.replace("_model.joblib", "")
        output_filename = f"inference__{model_base_name}_on_{test_dataset_name.replace('.csv', '')}__{timestamp}.csv"
        predictions_df.to_csv(os.path.join(OUTPUT_DIR, output_filename), index=False)
        
        return predictions_df.head(100), metrics_dict, f"Inference successful. Full results saved to:\n{output_filename}"
    except Exception as e:
        import traceback
        return None, {"Error": str(e)}, f"An error occurred during inference: {e}"

# --- Gradio UI Definition ---
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# Modular AI Model Trainer & Tester")
    
    with gr.Tabs():
        with gr.Tab("Training"):
            gr.Markdown("### Create and train new models.")
            with gr.Row():
                train_open_models_folder_btn, train_open_output_folder_btn, train_refresh_button = [gr.Button("📂 Open Saved Models"), gr.Button("📊 Open Outputs"), gr.Button("🔄 Refresh Lists")]
            gr.Markdown("### 1. Select Data and Model")
            with gr.Row():
                dataset_dd = gr.Dropdown(choices=discover_datasets(), label="Step 1: Select Training Dataset", interactive=True)
                load_columns_button = gr.Button("Step 2: Load Columns")
            with gr.Row():
                model_dd = gr.Dropdown(choices=list(discover_models().keys()), label="Model", interactive=True)
                device_selector = gr.Radio(choices=["CPU", "GPU"], value="CPU", label="Compute Device")
            gr.Markdown("### 2. Define Test Set")
            split_method_radio = gr.Radio(choices=["Auto-split (80/20)", "Select Test data csv", "Run on training set"], value="Auto-split (80/20)", label="Test Data Method")
            test_dataset_dd = gr.Dropdown(choices=discover_datasets(), label="Select Test Dataset File", interactive=True, visible=False)
            gr.Markdown("### 3. Configure Columns & Preprocessing")
            with gr.Row():
                target_column_dd = gr.Dropdown(label="Step 3: Select Target Column", interactive=False)
                encode_target_checkbox = gr.Checkbox(label="Encode Target Column (for string classes)", value=False, interactive=True)
                auto_identify_button = gr.Button("Step 4: Auto-Identify Features")
            with gr.Row():
                numerical_features_dd, scale_numericals_checkbox = [gr.Dropdown(label="Numerical Features", multiselect=True, interactive=False), gr.Checkbox(label="Scale Numericals?", value=True, interactive=True)]
            with gr.Row():
                onehot_features_dd, label_features_dd = [gr.Dropdown(label="One-Hot Encode", multiselect=True, interactive=False), gr.Dropdown(label="Label Encode", multiselect=True, interactive=False)]

            with gr.Accordion("Advanced: Edit Hyperparameters", open=False):
                hyperparameters_box = gr.Textbox(label="Model Hyperparameters (JSON format)", lines=4, visible=False, info="Edit the JSON string below.")

            train_run_button = gr.Button("Step 6: Run Training", variant="primary")
            gr.Markdown("## Training Outputs")
            with gr.Row():
                train_metrics_output, train_df_output = [gr.Label(label="Performance Metrics"), gr.DataFrame(label="Prediction Results (Sample)")]
            train_log_output = gr.Textbox(label="Run Status & Log", lines=4)

        with gr.Tab("Testing"):
            gr.Markdown("### Test a pre-existing, saved model on new data.")
            with gr.Row(): test_refresh_button = gr.Button("🔄 Refresh Saved Model List")
            saved_model_dd = gr.Dropdown(choices=discover_saved_models(), label="Select Saved Model", interactive=True)
            inference_dataset_dd = gr.Dropdown(choices=discover_datasets(), label="Select Test Dataset", interactive=True)
            inference_target_col = gr.Textbox(label="Target Column Name (Optional)", info="If your test set has a target column, enter its name here to calculate performance.")
            inference_run_button = gr.Button("Run Inference", variant="primary")
            gr.Markdown("## Inference Outputs")
            inference_metrics_output = gr.Label(label="Performance Metrics")
            inference_df_output = gr.DataFrame(label="Predictions (First 100 Rows)")
            inference_log_output = gr.Textbox(label="Run Status & Log", lines=4)

    # --- Connect UI components ---
    model_dd.change(fn=update_hyperparameters_ui, inputs=[model_dd], outputs=[hyperparameters_box])
    
    train_run_button.click(
        fn=train_and_evaluate,
        inputs=[model_dd, dataset_dd, device_selector, target_column_dd, 
                encode_target_checkbox, numerical_features_dd, onehot_features_dd, label_features_dd,
                scale_numericals_checkbox, hyperparameters_box,
                split_method_radio, test_dataset_dd],
        outputs=[train_df_output, train_metrics_output, train_log_output]
    )

    split_method_radio.change(fn=toggle_test_data_dropdown, inputs=[split_method_radio], outputs=[test_dataset_dd])
    train_refresh_button.click(fn=refresh_all_dropdowns, inputs=None, outputs=[dataset_dd, model_dd])
    load_columns_button.click(fn=load_target_columns, inputs=[dataset_dd], outputs=[target_column_dd])
    auto_identify_button.click(fn=update_numerical_dropdown, inputs=[dataset_dd, target_column_dd], outputs=[numerical_features_dd]
    ).then(fn=update_onehot_dropdown, inputs=[dataset_dd, target_column_dd], outputs=[onehot_features_dd]
    ).then(fn=update_label_dropdown, inputs=[dataset_dd, target_column_dd], outputs=[label_features_dd])
    
    train_open_models_folder_btn.click(lambda: open_folder(MODELS_OUTPUT_DIR), None, None)
    train_open_output_folder_btn.click(lambda: open_folder(OUTPUT_DIR), None, None)
    
    test_refresh_button.click(lambda: gr.update(choices=discover_saved_models()), inputs=None, outputs=saved_model_dd)
    inference_run_button.click(
        fn=run_inference,
        inputs=[saved_model_dd, inference_dataset_dd, inference_target_col],
        outputs=[inference_df_output, inference_metrics_output, inference_log_output]
    )

if __name__ == "__main__":
    app.launch()