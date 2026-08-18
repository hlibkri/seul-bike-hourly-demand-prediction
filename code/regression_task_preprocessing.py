import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


def drop_unnecessary_columns(df, columns: list = None, drop_date=True):
    if columns is None:
        columns = ["Kaggle_ID", "Record_id"]

    df = df.drop(columns=columns)
    if drop_date:
        df = df.drop(columns=["Date"])
    return df


def fix_season(df, season_column: str = "Seasons"):
    df[season_column] = (
        df[season_column]
        .str.replace(r".*A.*t.*m.*", "Autumn", regex=True)
        .str.replace(r".*Sp.*", "Spring", regex=True)
        .str.replace(r".*S.*m.*r.*", "Summer", regex=True)
        .str.replace(r".*W.*t.*r.*", "Winter", regex=True))
    return df


# Hour <0 bring no meaning and have no correlation with anything
# Wind speed over 8 can be high leverage point for some algorithms
# drop of >15 grad in seul is unrealistic
def filter_invalid(df, hour_column: str = "Hour", wind_column: str = "Wind speed", temp_change_column = "Temperature change 3h" ,temp_change_max = 15, include_engineered_features=True): #, max_wind: float = 8 
    df = df[df[hour_column] >= 0]
    df = df[df[wind_column] >= 0]
    if include_engineered_features:
        df = df[abs(df[temp_change_column]) <= temp_change_max]
    return df


def standardize(df, columns: list = None):
    if columns is None:
        columns = df.columns

    scaler = StandardScaler()

    df[columns] = scaler.fit_transform(df[columns])

    return df


class WeatherImputer(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        date_column="Date",
        features=None
    ):
        self.date_column = date_column
        self.features = features or [
            "Temperature",
            "Solar Radiation",
            "Rainfall",
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        order = X.sort_values([self.date_column, "Hour"]).index

        for feature in self.features:
            X.loc[order, feature] = (
                X.loc[order]
                .groupby(self.date_column)[feature]
                .transform(lambda x: x.interpolate().ffill().bfill())
            )

        X = X.drop(columns=self.date_column)
        return X

# =========================
# Feature engineering
# =========================


def _build_datetime(df, date_column: str = "Date", hour_column: str = "Hour"):
    dt = pd.to_datetime(df[date_column], dayfirst=True, errors="coerce")
    dt = dt + pd.to_timedelta(df[hour_column], unit="h")
    return dt


def add_dew_point_depression(
    df,
    temp_column: str = "Temperature",
    dew_point_column: str = "Dew point temperature",
    new_column: str = "Dew Point Depression",
):
    # How far the air temperature is above the dew point. Small values
    # (close to 0) mean the air is close to saturation.
    df[new_column] = df[temp_column] - df[dew_point_column]
    return df


def add_temperature_humidity_interaction(
    df,
    temp_column: str = "Temperature",
    humidity_column: str = "Humidity",
    new_column: str = "Temperature x Humidity",
):
    df[new_column] = df[temp_column] * df[humidity_column]
    return df


def add_wind_chill(
    df,
    temp_column: str = "Temperature",
    wind_column: str = "Wind speed",
    new_column: str = "Wind Chill",
):
    # Environment-Canada-style metric wind chill formula. It expects wind
    # speed in km/h, ours is in m/s, so convert first. The formula is only
    # meaningful for T <= 10C and wind > 4.8 km/h -- outside that range wind
    # chill is conventionally set equal to the air temperature.
    wind_kmh = (df[wind_column] * 3.6).clip(lower=0)

    wind_chill = (
        13.12
        + 0.6215 * df[temp_column]
        - 11.37 * wind_kmh ** 0.16
        + 0.3965 * df[temp_column] * wind_kmh ** 0.16
    )

    valid = (df[temp_column] <= 10) & (wind_kmh > 4.8)
    df[new_column] = np.where(valid, wind_chill, df[temp_column])
    return df


def add_cyclical_hour(
    df,
    hour_column: str = "Hour",
    sin_column: str = None,
    cos_column: str = None,
):
    # Encode hour-of-day as a point on a circle so that hour 23 and hour 0
    # are close together, unlike the raw integer. The raw Hour column is
    # left untouched/kept in the frame.
    if sin_column is None:
        sin_column = f"{hour_column} sin"
    if cos_column is None:
        cos_column = f"{hour_column} cos"

    radians = 2 * np.pi * df[hour_column] / 24
    df[sin_column] = np.sin(radians)
    df[cos_column] = np.cos(radians)
    return df


def add_fog_risk(
    df,
    temp_column: str = "Temperature",
    dew_point_column: str = "Dew point temperature",
    humidity_column: str = "Humidity",
    new_column: str = "Fog Risk",
    depression_threshold: float = 2.5,
    humidity_threshold: float = 90,
):
    # Fog tends to form when air is near saturation: temperature close to
    # dew point AND high relative humidity.
    depression = df[temp_column] - df[dew_point_column]
    df[new_column] = (
        (depression <= depression_threshold) & (df[humidity_column] >= humidity_threshold)
    ).astype(int)
    return df


def add_recent_rainfall(
    df,
    date_column: str = "Date",
    hour_column: str = "Hour",
    rainfall_column: str = "Rainfall",
    window: int = 3,
    new_column: str = None,
):
    # Sum of rainfall over the current hour and the (window - 1) hours
    # before it, in chronological order.
    if new_column is None:
        new_column = f"Rainfall accumulation {window}h"

    dt = _build_datetime(df, date_column, hour_column)
    order = dt.sort_values(kind="stable").index

    rolled = (
        df.loc[order, rainfall_column]
        .rolling(window=window, min_periods=1)
        .sum()
    )

    df[new_column] = rolled.reindex(df.index)
    return df


def add_temperature_change(
    df,
    date_column: str = "Date",
    hour_column: str = "Hour",
    temp_column: str = "Temperature",
    periods: int = 3,
    new_column: str = None,
):
    # Difference between the current temperature and the temperature
    # `periods` hours earlier, in chronological order.
    if new_column is None:
        new_column = f"Temperature change {periods}h"

    dt = _build_datetime(df, date_column, hour_column)
    order = dt.sort_values(kind="stable").index

    diff = df.loc[order, temp_column].diff(periods=periods)

    df[new_column] = diff.reindex(df.index).fillna(0)
    return df


def add_rolling_averages(
    df,
    date_column: str = "Date",
    hour_column: str = "Hour",
    columns: tuple = ("Temperature", "Wind speed"),
    window: int = 3,
    suffix: str = None,
):
    # Rolling mean over the current hour and the (window - 1) hours before
    # it, in chronological order, for each column requested.
    if suffix is None:
        suffix = f" rolling {window}h avg"

    dt = _build_datetime(df, date_column, hour_column)
    order = dt.sort_values(kind="stable").index

    for col in columns:
        rolled = df.loc[order, col].rolling(window=window, min_periods=1).mean()
        df[f"{col}{suffix}"] = rolled.reindex(df.index)

    return df


def engineer_features(
    df,
    date_column: str = "Date",
    hour_column: str = "Hour",
    rainfall_window: int = 3,
    temperature_change_periods: int = 3,
    rolling_window: int = 3,
    rolling_columns: tuple = ("Temperature", "Wind speed"),
):
    """
    Convenience wrapper that applies all engineered features at once.
    Requires `date_column` to still be present in `df` (call this before
    dropping the Date column).
    """
    df = add_dew_point_depression(df)
    df = add_temperature_humidity_interaction(df)
    df = add_wind_chill(df)
    df = add_cyclical_hour(df, hour_column=hour_column)
    df = add_fog_risk(df)
    df = add_recent_rainfall(df, date_column=date_column, hour_column=hour_column, window=rainfall_window)
    df = add_temperature_change(df, date_column=date_column, hour_column=hour_column, periods=temperature_change_periods)
    df = add_rolling_averages(df, date_column=date_column, hour_column=hour_column, columns=rolling_columns, window=rolling_window)
    return df