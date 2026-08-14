import re

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def drop_unnecessary(df, columns: list = None):
    if columns is None:
        columns = ["Kaggle_ID", "Record_id"]

    df = df.drop(columns=columns)
    return df

def fix_season(df, season_column: str = "Seasons"):
    df[season_column] = (
        df[season_column]
        .str.replace(r".*A.*t.*m.*", "Autumn", regex=True)
        .str.replace(r".*Sp.*", "Spring", regex=True)
        .str.replace(r".*S.*m.*r.*", "Summer", regex=True)
        .str.replace(r".*W.*t.*r.*", "Winter", regex=True))
    return df

def filter_invalid_hours(df, hour_column: str = "Hour"):
    df = df[df[hour_column] >= 0]
    return df

def transform_categorical(df):
    categorical_cols = df.select_dtypes(include=["object", "str"]).columns
    numerical_cols = df.select_dtypes(include=["number"]).columns

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    df_encoded = encoder.fit_transform(df[categorical_cols])

    df_encoded = pd.DataFrame(
        df_encoded,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=df.index
    )

    df = pd.concat(
        [df[numerical_cols], df_encoded],
        axis=1
    )

    return df


def standardize(df, columns: list = None):
    if columns is None:
        columns = df.columns

    scaler = StandardScaler()

    df[columns] = scaler.fit_transform(df[columns])

    return df   