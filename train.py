import os
import warnings
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.datasets import load_wine
import mlflow
import mlflow.sklearn
import joblib
import subprocess

# --- Configuration ---
warnings.filterwarnings("ignore")
mlflow.set_tracking_uri("./mlruns")
EXPERIMENT_NAME = "Wine Classification Pipeline"

try:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
    print(f"Created new experiment with ID: {experiment_id}")
except mlflow.exceptions.MlflowException as e:
    if "RESOURCE_ALREADY_EXISTS" in str(e):
        experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
        print(f"Experiment '{EXPERIMENT_NAME}' already exists with ID: {experiment_id}")
    else:
        raise e 

mlflow.set_experiment(experiment_name=EXPERIMENT_NAME) 

# ---  Log Training Script Hash For tag ---
def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "N/A"

# --- Functions (load_data, preprocess_data, evaluate_model) ---
def load_data():
    print("--- Loading Data ---")
    wine = load_wine()
    data = pd.DataFrame(data=np.c_[wine['data'], wine['target']],
                        columns=wine['feature_names'] + ['target'])
    X = data.drop('target', axis=1)
    y = data['target']
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print("Dataset Source: Scikit-learn built-in datasets")
    print("Dataset Version: load_wine (sklearn version specific)")
    print("-" * 20)
    dataset_info = {
        "source": "Scikit-learn built-in load_wine",
        "sklearn_version": pd.__version__,
        "num_samples": X.shape[0],
        "num_features": X.shape[1],
        "target_classes": len(np.unique(y))
    }
    return X, y, dataset_info

def preprocess_data(X, y, test_size=0.3, random_state=42):
    print("--- Data Preprocessing ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    print(f"Train set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Data scaled using StandardScaler (fit on train, transform train/test)")
    print("-" * 20)
    
    scaler_path = "src/model/scaler.joblib"
    joblib.dump(scaler, scaler_path)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler_path 

def train_and_tune_model(model_name, model_instance, param_grid, X_train, y_train, cv=5):
    print(f"--- Training and validating Model [{model_name}] with GridSearchCV ---")

    grid_search = GridSearchCV(estimator=model_instance, param_grid=param_grid, cv=cv, n_jobs=-1, scoring='accuracy')
    print(f"Performing GridSearchCV for {model_name} with parameters: {param_grid}")
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_model = grid_search.best_estimator_

    print(f"Best parameters found for {model_name}: {best_params}")
    print(f"Best cross-validation accuracy for {model_name}: {best_score:.4f}")
    print("-" * 20)
    return best_model, best_params, best_score

def evaluate_model(model_name, model, X_test, y_test):
    print(f"--- Evaluating Model [{model_name}] ---")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Test Accuracy ({model_name}): {accuracy:.4f}")
    print(f"Test Precision (weighted, {model_name}): {precision:.4f}")
    print(f"Test Recall (weighted, {model_name}): {recall:.4f}")
    print(f"Test F1-score (weighted, {model_name}): {f1:.4f}")

    report = classification_report(y_test, y_pred, target_names=load_wine().target_names)
    print(f"\nClassification Report ({model_name}):\n", report)
    print("-" * 20)

    metrics = {
        "test_accuracy": accuracy,
        "test_precision_weighted": precision,
        "test_recall_weighted": recall,
        "test_f1_weighted": f1
    }
    return metrics, report

def log_to_mlflow(model_name, dataset_info, grid_search_params, best_params, cv_score, test_metrics, model, scaler_path, report_str):
    if mlflow.active_run():
        mlflow.end_run()

    print(f"--- Logging to MLflow for [{model_name}] ---")
    
    with mlflow.start_run(run_name=f"{model_name}_GridSearch") as run:
        mlflow.set_tag("git_commit", get_git_commit())
        
        run_id = run.info.run_id
        print(f"MLflow Run ID for {model_name}: {run_id}")
        print(f"MLflow Artifact URI: {mlflow.get_artifact_uri()}")

        # 1. Log Dataset Information 
        if run.info.run_uuid == mlflow.search_runs(experiment_ids=[run.info.experiment_id]).iloc[0]['run_id']: # Chỉ log dataset ở run đầu tiên
             print("Logging dataset info (first run only)...")
             mlflow.log_params(dataset_info)

        # 2. Log Hyperparameters
        print("Logging hyperparameters...")
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("tuning_method", "GridSearchCV")
        mlflow.log_params({"cv_folds": grid_search_params['cv']})
        mlflow.log_param("grid_search_space", str(grid_search_params['param_grid']))
        mlflow.log_params(best_params) # Log các tham số tốt nhất tìm được cho model này

        # 3. Log Metrics
        print("Logging metrics...")
        mlflow.log_metric("best_cv_accuracy", cv_score)
        mlflow.log_metrics(test_metrics)

        # 4. Log Model (Checkpoint)
        print("Logging model...")
        mlflow.sklearn.log_model(model, f"model_{model_name}")

        # 5. Log Preprocessing Artifact (Scaler)
        print("Logging scaler...")
        mlflow.log_artifact(scaler_path, artifact_path="preprocessing")

        # 6. Log Classification Report
        print("Logging classification report...")
        mlflow.log_text(report_str, f"evaluation/classification_report_{model_name}.txt")

        # 7. Log Source Code 
        if run.info.run_uuid == mlflow.search_runs(experiment_ids=[run.info.experiment_id]).iloc[0]['run_id']:
            print("Logging source code (first run only)...")
            mlflow.log_artifact(__file__, artifact_path="code")

        print("-" * 20)
        print(f"MLflow logging completed for {model_name}.")

if __name__ == "__main__":
    print("===== Starting ML Pipeline - Multi-Algorithm Comparison =====")

    # --- Load Data ---
    X, y, dataset_info_dict = load_data()

    # --- Preprocess Data ---
    X_train_scaled, X_test_scaled, y_train, y_test, scaler_filepath = preprocess_data(X, y)

    # --- Define Models and Hyperparameter Grids ---
    cv_folds = 5 # Number of cross-validation folds
    models_to_run = {
        "LogisticRegression": {
            "model": LogisticRegression(solver='liblinear', random_state=42, max_iter=1000), 
            "param_grid": {'C': [0.1, 1.0, 10.0, 100.0], 'penalty': ['l1', 'l2']}
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "param_grid": {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
        },
        "SVM": {
            "model": SVC(probability=True, random_state=42), 
            "param_grid": {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto', 0.1, 1.0]}
        }
    }

    # --- Loop through each model configuration ---
    for model_name, config in models_to_run.items():
        print(f"\n===== Processing Model: {model_name} =====")

        # --- Train Model with Tuning ---
        best_model_found, best_params_found, best_cv_score = train_and_tune_model(
            model_name=model_name,
            model_instance=config["model"],
            param_grid=config["param_grid"],
            X_train=X_train_scaled,
            y_train=y_train,
            cv=cv_folds
        )

        # --- Evaluate Model ---
        test_metrics_dict, report_text = evaluate_model(
            model_name=model_name,
            model=best_model_found,
            X_test=X_test_scaled,
            y_test=y_test
        )

        # --- Log to MLflow ---
        log_to_mlflow(
            model_name=model_name,
            dataset_info=dataset_info_dict,
            grid_search_params={'param_grid': config["param_grid"], 'cv': cv_folds},
            best_params=best_params_found,
            cv_score=best_cv_score,
            test_metrics=test_metrics_dict,
            model=best_model_found,
            scaler_path=scaler_filepath, 
            report_str=report_text
        )
        
    # --- Save Model for API ---
    api_model_path = f"src/model/wine_classification.pkl"
    joblib.dump(best_model_found, api_model_path)
    print(f"Saved model for {model_name} to: {api_model_path}")
