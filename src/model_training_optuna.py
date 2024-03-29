import datetime
import time
import warnings

import mlflow
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    confusion_matrix,
)

from sklearn.model_selection import train_test_split

from experiment_tracking import (
    champion_callback,
    get_or_create_experiment,
    plot_feature_importance,
)
from feature_engineering import create_all_features
from preprocessing import combine_train_and_original_dfs, preprocess_df

warnings.filterwarnings("ignore")
target = "NObeyesdad"
random_state = 0
n_trials = 25
current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
run_name = f"{current_datetime}_lgbm_add_log_physical_{n_trials}"

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
            "verbose_eval": -1,
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

train = combine_train_and_original_dfs(df_train, df_original)
X_test = df_test.drop(["id"], axis=1)

# Apply preprocessing
train, X_test = preprocess_df(train), preprocess_df(X_test)

X = train.drop([target], axis=1)
y = train[target]

# Train-test split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=random_state
)

# Apply feature engineering
train, X_test = create_all_features(train), create_all_features(X_test)

# Initiate the parent run and call the hyperparameter tuning child run logic
with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):
    start_time = time.time()

    # Initialize the Optuna study
    study = optuna.create_study(direction="maximize")

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

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_valid, y_pred, average="weighted"
    )
    conf_matrix = confusion_matrix(y_valid, y_pred)
    report = classification_report(y_valid, y_pred, output_dict=True)

    # Log the feature importances plot
    importances = plot_feature_importance(model, X=X)
    mlflow.log_figure(figure=importances, artifact_file="feature_importances.png")

    # Log the confusion matrix as an artifact (convert it to a file first)
    np.savetxt("confusion_matrix.csv", conf_matrix, delimiter=",")
    mlflow.log_artifact("confusion_matrix.csv")

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv("classification_report.csv")
    mlflow.log_artifact("classification_report.csv")

    artifact_path = "model"

    mlflow.lightgbm.log_model(
        lgb_model=model,
        artifact_path=artifact_path,
        input_example=X_train.iloc[[0]],
        metadata={"model_data_version": 1},
    )
    end_time = time.time()
    duration = end_time - start_time
    duration_str = str(datetime.timedelta(seconds=duration))
    mlflow.log_metric("Duration in seconds", duration)
    mlflow.log_param("Readable Duration", duration_str)
    print(f"Experiment duration: {duration_str}")
    print("Best accuracy:", study.best_value)
    print("Best params:", study.best_params)

    # Get the logged model uri so that we can load it from the artifact store
    model_uri = mlflow.get_artifact_uri(artifact_path)
    print("model_uri:", model_uri)

    mlflow.end_run()
