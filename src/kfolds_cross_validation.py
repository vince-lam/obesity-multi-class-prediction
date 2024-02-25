import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/processed/train_folds.csv")
df_test = pd.read_csv("data/raw/test.csv")
submission = pd.read_csv("data/raw/sample_submission.csv")

random_state = 0
target = "NObeyesdad"

params = {
    "objective": "multiclass",  # Objective function for the model
    "metric": "multi_logloss",  # Evaluation metric
    "verbosity": -1,  # Verbosity level (-1 for silent)
    "boosting_type": "gbdt",  # Gradient boosting type
    "random_state": 0,  # Random state for reproducibility
    "num_class": 7,  # Number of classes in the dataset
    "learning_rate": 0.012895148872894106,  # Learning rate for gradient boosting
    "n_estimators": 457,  # Number of boosting iterations
    "lambda_l1": 0.008179204992451842,  # L1 regularization term
    "lambda_l2": 0.022669285325135756,  # L2 regularization term
    "max_depth": 12,  # Maximum depth of the trees
    "colsample_bytree": 0.4240998469631964,  # Fraction of features to consider for each tree
    "subsample": 0.9658150639983177,  # Fraction of samples to consider for each boosting iteration
    "min_child_samples": 46,  # Minimum number of data needed in a leaf
    "njobs": -1,  # Number of parallel threads
}

lgbm_classifier = LGBMClassifier(**params)


useful_features = [c for c in df.columns if c not in ("id", target, "kfold")]
categorical_columns = df.select_dtypes(include=[object]).columns.tolist()
categorical_columns.remove(target)
df_test = df_test[useful_features]


def run_kfolds_cross_validation(
    model,
    n_folds=5,
    df=df,
    df_test=df_test,
    target=target,
    useful_features=useful_features,
    categorical_columns=categorical_columns,
):
    final_predictions = []
    accuracy_scores = []

    for fold in range(n_folds):
        x_train = df[df.kfold != fold].reset_index(drop=True)
        x_valid = df[df.kfold == fold].reset_index(drop=True)
        x_test = df_test.copy()

        y_train = x_train[target]
        y_valid = x_valid[target]

        x_train = x_train[useful_features]
        x_valid = x_valid[useful_features]

        # Apply one-hot encoding to the categorical columns
        x_train = pd.get_dummies(x_train, columns=categorical_columns, drop_first=True)
        x_valid = pd.get_dummies(x_valid, columns=categorical_columns, drop_first=True)
        x_test = pd.get_dummies(x_test, columns=categorical_columns, drop_first=True)

        # Ensure all the columns in the test set are also present in the train set
        x_test, x_train = x_test.align(x_train, join="left", axis=1)
        x_test, x_valid = x_test.align(x_valid, join="left", axis=1)
        x_train.fillna(0, inplace=True)
        x_valid.fillna(0, inplace=True)

        model.fit(x_train, y_train)

        preds_valid = model.predict(x_valid)
        preds_test = model.predict(x_test)

        final_predictions.append(preds_test)
        accuracy_scores.append(accuracy_score(y_valid, preds_valid))
        print(fold, accuracy_score(y_valid, preds_valid))

    return final_predictions, accuracy_scores


final_predictions, accuracy_scores = run_kfolds_cross_validation(model=lgbm_classifier)

mean_accuracy = np.mean(accuracy_scores)
print(f"Mean accuracy: {mean_accuracy}")

# Get mode of predictions from the folds
transposed_predictions = list(map(list, zip(*final_predictions)))
final_mode_predictions = []
for predictions in transposed_predictions:
    # Use np.unique to count occurrences of each category and find the mode
    values, counts = np.unique(predictions, return_counts=True)
    index = np.argmax(counts)  # Index of the most frequent element
    mode_prediction = values[index]  # The most frequent element
    final_mode_predictions.append(mode_prediction)

submission[target] = final_mode_predictions
submission.to_csv(
    "data/submissions/submission3_lgbm_no_feature_eng_kfolds.csv", index=False
)
