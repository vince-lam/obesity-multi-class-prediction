import mlflow
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from feature_engineering import create_bmi_features
from preprocessing import (
    get_numerical_columns,
    get_categorical_columns,
    combine_train_and_original_dfs,
)
from experiment_tracking import (
    get_or_create_experiment,
    champion_callback,
    plot_feature_importance,
)

target = "NObeyesdad"
random_state = 0
n_trials = 20
run_name = f"1_optuna_lgbm_bmi_feats_{n_trials}"

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
experiment_id = get_or_create_experiment("Obesity Prediction")
# Set the current active MLflow experiment
mlflow.set_experiment(experiment_id=experiment_id)
# override Optuna's default logging to ERROR only
optuna.logging.set_verbosity(optuna.logging.ERROR)


def objective(trial):
    with mlflow.start_run(nested=True):
        # Define parameters to be optimized for the LGBMClassifier
        params = {
            "objective": "multiclass",
            "metric": "multi_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "random_state": random_state,
            "num_class": 7,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05),
            "n_estimators": trial.suggest_int("n_estimators", 400, 600),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.005, 0.015),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.02, 0.06),
            "max_depth": trial.suggest_int("max_depth", 6, 14),
            "num_leaves": trial.suggest_int("num_leaves", 2**6, 2**14),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.9),
            "subsample": trial.suggest_float("subsample", 0.8, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        }

        # Train LightGBM model
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)
        score = model.score(X_valid, y_valid)

        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("Mean accuracy", score)

    return score


# Load data
df_train = pd.read_csv("data/raw/train.csv")
df_test = pd.read_csv("data/raw/test.csv")
df_original = pd.read_csv("data/raw/ObesityDataSet.csv")

# Preprocessing
numerical_cols = get_numerical_columns(df_train)
categorical_cols = get_categorical_columns(df_train)

train = combine_train_and_original_dfs(df_train, df_original)
X_test = df_test.drop(["id"], axis=1)

train = create_bmi_features(train)
X_test = create_bmi_features(X_test)

# Apply one-hot encoding
train = pd.get_dummies(train, columns=categorical_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

X = train.drop([target], axis=1)
y = train[target]

# Train-test split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=random_state
)

# Initiate the parent run and call the hyperparameter tuning child run logic
with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):
    # Initialize the Optuna study
    study = optuna.create_study(direction="minimize")

    # Execute the hyperparameter optimization trials.
    # Note the addition of the `champion_callback` inclusion to control our logging
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[champion_callback],
    )

    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_accuracy", study.best_value)

    # Log tags
    mlflow.set_tags(
        tags={
            "project": "Obesity Prediction Project",
            "optimizer_engine": "optuna",
            "model_family": "lightGBM",
            "feature_set_version": 1,
        }
    )

    # Log a fit model instance
    model = LGBMClassifier(**study.best_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)

    # Log the feature importances plot
    importances = plot_feature_importance(model, X=X)
    mlflow.log_figure(figure=importances, artifact_file="feature_importances.png")

    artifact_path = "model"

    mlflow.lightgbm.log_model(
        lgb_model=model,
        artifact_path=artifact_path,
        input_example=X_train.iloc[[0]],
        metadata={"model_data_version": 1},
    )

    print("Best params:", study.best_params)

    # Get the logged model uri so that we can load it from the artifact store
    model_uri = mlflow.get_artifact_uri(artifact_path)
    print("model_uri:", model_uri)
