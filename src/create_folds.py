import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df_train = pd.read_csv("data/raw/train.csv")
    df_original = pd.read_csv("data/raw/ObesityDataSet.csv")
    random_state = 0
    num_folds = 5
    output_csv_path = "data/processed/train_and_original_folds.csv"

    train = pd.concat([df_train, df_original]).drop(["id"], axis=1).drop_duplicates()
    train = train.reset_index(drop=True)

    kf = model_selection.KFold(
        n_splits=num_folds, shuffle=True, random_state=random_state
    )

    for fold, (train_indices, valid_indices) in enumerate(kf.split(X=train)):
        train.loc[valid_indices, "kfold"] = fold

    train.to_csv(output_csv_path, index=False)
