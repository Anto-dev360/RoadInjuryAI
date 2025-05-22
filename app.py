"""
app.py

Application main file.

Author: Anthony Morin
Created: 2025-05-21
Project: RoadInjuryAI
License: MIT
"""
from scripts.data_preprocessing import (load_data,
                               clean_data_pipeline,
                               data_encoding)
from scripts.model_training import random_forest, xg_boost


def main():
    print("⏳ Loading and cleaning data...")
    df = load_data()
    df_clean = clean_data_pipeline(df)
    df_encoded, X_train_data = data_encoding(df_clean)

    # Train models and display evaluation metrics
    print("\n⏳ Training Random Forest model...")
    random_forest(X_train_data, df_encoded['grav'])
    print("\n⏳ Training Random Gradient Boosting classifier...")
    xg_boost(X_train_data, df_encoded['grav'])


if __name__ == "__main__":
    main()