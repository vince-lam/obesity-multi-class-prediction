import matplotlib.pyplot as plt
import mlflow
import experiment_tracking
import pandas as pd
import seaborn as sns


def get_or_create_experiment(experiment_name):
    """
    Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

    This function checks if an experiment with the given name exists within MLflow.
    If it does, the function returns its ID. If not, it creates a new experiment
    with the provided name and returns its ID.

    Parameters:
    - experiment_name (str): Name of the MLflow experiment.

    Returns:
    - str: ID of the existing or newly created MLflow experiment.
    """

    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)


# define a logging callback that will report on only new challenger parameter configurations if a
# trial has usurped the state of 'best conditions'


def champion_callback(study, frozen_trial):
    """
    Logging callback that will report when a new trial iteration improves upon existing
    best trial values.

    Note: This callback is not intended for use in distributed computing systems such as Spark
    or Ray due to the micro-batch iterative implementation for distributing trials to a cluster's
    workers or agents.
    The race conditions with file system state management for distributed trials will render
    inconsistent values with this callback.
    """

    winner = study.user_attrs.get("winner", None)

    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            improvement_percent = (
                abs(winner - study.best_value) / study.best_value
            ) * 100
            print(
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                f"{improvement_percent: .4f}% improvement"
            )
        else:
            print(
                f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}"
            )


def plot_feature_importance(model, X):
    """
    Plots feature importance for a LightGBM model.

    Args:
    - model: A trained LightGBM model

    Returns:
    - fig: The matplotlib figure object
    """

    fig, ax = plt.subplots(figsize=(12, 10))

    feature_importance = model.feature_importances_
    feature_importance_df = pd.DataFrame(
        {"Feature": X.columns, "Importance": feature_importance}
    )
    feature_importance_df = feature_importance_df.sort_values(
        by="Importance", ascending=False
    )

    sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    sns.despine(left=True, bottom=True)

    return fig
