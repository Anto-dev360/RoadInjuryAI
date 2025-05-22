"""
data_preprocessing.py

This module contains functions for preparing the road accident dataset before training.

It includes:
- Dataset loading
- Column filtering and NaN cleaning
- Hour discretization and geospatial clustering
- One-hot encoding of categorical features

Used for transforming raw merged datasets into ML-ready formats.

Author: Anthony Morin
Created: 2025-05-21
Project: RoadInjuryAI
License: MIT
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

from pathlib import Path
from config.settings import DATA_DIR_PATH

def load_data(data_dir=DATA_DIR_PATH):
    """
    Load and merge accident-related datasets.

    This function reads four CSV files:
    - Characteristics of accidents
    - Accident locations
    - Vehicles involved
    - Victims involved

    It merges them into a single DataFrame with full accident information
    by joining on `Num_Acc` and `num_veh`.

    Args:
        data_dir (str): Path to the folder containing the CSV files.

    Returns:
        pd.DataFrame: Merged DataFrame containing victim, vehicle, and accident details.
    """
    data_path = Path(data_dir)

    # Load CSV files
    df_characteristics = pd.read_csv(data_path / "carac.csv", sep=';')
    df_vehicles        = pd.read_csv(data_path / "veh.csv", sep=';')
    df_victims         = pd.read_csv(data_path / "vict.csv", sep=';')
    # Mixed-type columns (e.g., 'voie', 'v2') are tolerated here as they
    # will be excluded later in favor of geolocation features
    df_locations       = pd.read_csv(data_path / "lieux.csv", sep=';', dtype={'voie': str, 'v2': str})

    # Merge victims with vehicles
    df_victim_vehicle = df_victims.merge(df_vehicles, on=["Num_Acc", "num_veh"], how="left")

    # Merge accident characteristics and locations
    df_accident = df_characteristics.merge(df_locations, on="Num_Acc", how="left")

    # Final merged dataset
    df_final = df_victim_vehicle.merge(df_accident, on="Num_Acc", how="left")

    return df_final


def drop_useless_columns(df):
    """
    Drop columns that are mostly empty or not useful for prediction.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with selected columns removed.
    """
    columns_to_drop = [
        'v1', 'v2', 'lartpc', 'larrout', 'locp', 'etatp',
        'actp', 'voie', 'pr1', 'pr', 'place'
    ]
    return df.drop(columns=[col for col in columns_to_drop if col in df.columns])


def drop_missing_rows(df):
    """
    Drop rows that contain any missing values.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame without NaN rows.
    """
    return df.dropna()


def clean_data_pipeline(df):
    """
    Apply a full data cleaning pipeline to the accident dataset.

    The cleaning steps include:
    1. Removing specific columns with over 10% missing values.
    2. Removing rows that contain any remaining NaNs.

    Args:
        df (pd.DataFrame): Raw merged dataset.

    Returns:
        pd.DataFrame: Cleaned dataset ready for preprocessing.
    """
    df_clean = df.copy()
    df_clean = drop_useless_columns(df_clean)
    df_clean = drop_missing_rows(df_clean)
    return df_clean


def data_encoding(df):
    """
    Perform encoding and feature engineering on the accident dataset.

    Steps:
    1. Discretize hour ('hrmn') into 24 bins.
    2. Cluster geographic coordinates (lat, long) into 15 zones.
    3. One-hot encode selected categorical features for modeling.

    Args:
        df (pd.DataFrame): Cleaned dataset.

    Returns:
        tuple:
            - pd.DataFrame: Original DataFrame with added 'geo' and transformed 'hrmn'.
            - pd.DataFrame: Feature matrix (X) with one-hot encoded variables.
    """
    # Discretize hour feature
    df['hrmn'] = pd.cut(df['hrmn'], 24, labels=[str(i) for i in range(0, 24)]).astype(str)

    # Create spatial clusters
    X_cluster = np.array(list(zip(df['lat'], df['long'])))
    clustering = KMeans(n_clusters=15, random_state=0).fit(X_cluster)
    df['geo'] = clustering.labels_

    # One-hot encoding of selected features => ease model training
    features = [
        'catu','sexe','trajet','secu','catv','an_nais','mois','occutc',
        'obs','obsm','choc','manv','lum','agg','int','atm','col','gps',
        'catr','circ','vosp','prof','plan','surf','infra','situ','hrmn','geo'
    ]
    X_train_data = pd.get_dummies(df[features].astype(str))

    return df, X_train_data
