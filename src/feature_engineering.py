import numpy as np
import pandas as pd


def create_bmi_features(df):
    tmp = df.copy()
    tmp["bmi"] = tmp["Weight"] / tmp["Height"] ** 2
    tmp["bmi_prime"] = tmp["bmi"] / 25
    tmp["bmi_times_faf"] = tmp["bmi"] * tmp["FAF"]

    return tmp


def create_height_weight_ratio(df):
    tmp = df.copy()
    tmp["height_weight_ratio"] = df["Height"] / df["Weight"]
    return tmp


def encode_eating_habits(df):
    tmp = df.copy()
    tmp["Eating_Habits_Score"] = tmp["FAVC"] * 2 + tmp["NCP"] + tmp["CAEC_Ordinal"]
    return tmp


def create_water_intake(df):
    # Assuming 2 liters as adequate
    tmp = df.copy()
    tmp["Adequate_Water_Intake"] = tmp["CH2O"].apply(lambda x: 1 if x >= 2 else 0)
    return tmp


def create_physical_score(df):
    tmp = df.copy()
    tmp["Physical_Activity_Score"] = tmp["FAF"] * (1 - tmp["TUE"])
    return tmp


def create_interaction_features(df):
    tmp = df.copy()
    tmp["Age_FAF_Interact"] = tmp["Age"] * tmp["FAF"]
    tmp["Height_Weight_Interact"] = tmp["Height"] * tmp["Weight"]
    return tmp


# Transform skewed features
def create_log_feats(df):
    tmp = df.copy()
    tmp["FAF_Log"] = tmp["FAF"].apply(lambda x: np.log(x + 1))
    tmp["TUE_Log"] = tmp["TUE"].apply(lambda x: np.log(x + 1))
    return tmp


def create_all_features(df):
    tmp = df.copy()
    tmp = create_bmi_features(tmp)
    tmp = create_height_weight_ratio(tmp)
    tmp = encode_eating_habits(tmp)
    tmp = create_water_intake(tmp)
    tmp = create_interaction_features(tmp)
    tmp = create_physical_score(tmp)
    tmp = create_log_feats(tmp)
    return tmp
