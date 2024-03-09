import datetime
import warnings

import mlflow
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from experiment_tracking import get_or_create_experiment
from feature_engineering import create_all_features
from preprocessing import combine_train_and_original_dfs, preprocess_df

warnings.filterwarnings("ignore")
target = "NObeyesdad"
random_state = 0
current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
run_name = f"{current_datetime}_single_add_feats"

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
experiment_id = get_or_create_experiment("Obesity Prediction")
# Set the current active MLflow experiment
mlflow.set_experiment(experiment_id=experiment_id)

# Load data
df_train = pd.read_csv("data/raw/train.csv")
df_test = pd.read_csv("data/raw/test.csv")
df_original = pd.read_csv("data/raw/ObesityDataSet.csv")
submission = pd.read_csv("data/raw/sample_submission.csv")


train = combine_train_and_original_dfs(df_train, df_original)
X_test = df_test.drop(["id"], axis=1)

X = train.drop([target], axis=1)
y = train[target]

# Train-test split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=random_state
)

# Apply preprocessing
X_train, X_valid, X_test = (
    preprocess_df(X_train),
    preprocess_df(X_valid),
    preprocess_df(X_test),
)
X_train, X_valid, X_test = (
    create_all_features(X_train),
    create_all_features(X_valid),
    create_all_features(X_test),
)

params = {
    "objective": "multiclass",  # Objective function for the model
    "metric": "multi_logloss",  # Evaluation metric
    "verbosity": -1,  # Verbosity level (-1 for silent)
    "boosting_type": "gbdt",  # Gradient boosting type
    "random_state": 0,  # Random state for reproducibility
    "num_class": 7,  # Number of classes in the dataset
    "learning_rate": 0.02316776970107541,
    "n_estimators": 587,
    "lambda_l1": 0.006671527147173783,
    "lambda_l2": 0.03883176994133525,
    "max_depth": 6,
    "num_leaves": 1552,
    "colsample_bytree": 0.4513594001073964,
    "subsample": 0.851952872001092,
    "min_child_samples": 48,
    "njobs": -1,  # Number of parallel threads
}

# Initiate the parent run and call the hyperparameter tuning child run logic
with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):

    mlflow.log_params(params)
    # Log tags
    mlflow.set_tags(
        tags={
            "project": "Obesity Prediction Project",
            "model_family": "lightGBM",
            "feature_set_version": 1,
        }
    )
    artifact_path = "model"

    # Log a fit model instance
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)

    mlflow.lightgbm.log_model(
        lgb_model=model,
        artifact_path=artifact_path,
        input_example=X_train.iloc[[0]],
        metadata={"model_data_version": 1},
    )
    accuracy = accuracy_score(y_valid, y_pred)
    mlflow.log_metric("best_accuracy", accuracy)
    print("Accuracy:", accuracy)

    # Get the logged model uri so that we can load it from the artifact store
    model_uri = mlflow.get_artifact_uri(artifact_path)
    print("model_uri:", model_uri)

# Submission
predictions = model.predict(X_test)
submission["NObeyesdad"] = predictions
output_dir = "outputs/submissions/"
filename_suffix = "best_optuna"
current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
filename = f"{output_dir}{current_datetime}_{filename_suffix}.csv"
submission.to_csv(filename, index=False)
