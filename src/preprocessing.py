import numpy as np
import pandas as pd


def get_numerical_columns(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_columns(df):
    cat_cols = df.select_dtypes(include=[object]).columns.tolist()
    return cat_cols


def combine_train_and_original_dfs(df_train, df_original):
    train = pd.concat([df_train, df_original]).drop(["id"], axis=1).drop_duplicates()
    return train


def convert_yes_no_to_binary(df, columns):
    """
    Converts specified columns in a DataFrame from 'yes'/'no' to binary (1/0).

    Parameters:
    - df: pandas DataFrame containing the data.
    - columns: list of column names to convert.

    Returns:
    - DataFrame with specified columns converted to binary.
    """
    df[columns] = df[columns].replace({"yes": 1, "no": 0})
    return df


def apply_ordinal_encoding(df):
    # Ordinal Encoding for Ordered Categories
    tmp = df.copy()
    caec_order = {"Never": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    calc_order = {"no": 0, "Sometimes": 1, "Frequently": 2}
    tmp["CAEC_Ordinal"] = tmp["CAEC"].map(caec_order)
    tmp["CALC_Ordinal"] = tmp["CALC"].map(calc_order)

    tmp = tmp.drop(["CAEC", "CALC"], axis=1)

    return tmp


# Apply one hot encoding
def apply_ohe(df):
    tmp = df.copy()
    tmp = pd.get_dummies(tmp, columns=["Gender", "MTRANS"], drop_first=True)

    return tmp


def preprocess_df(df):
    tmp = df.copy()
    binary_cols = ["family_history_with_overweight", "FAVC", "SMOKE", "SCC"]
    tmp = convert_yes_no_to_binary(tmp, binary_cols)
    tmp = apply_ordinal_encoding(tmp)
    tmp = apply_ohe(tmp)

    return tmp
