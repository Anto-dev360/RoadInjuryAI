"""
app.py

Application main file.

Author: Anthony Morin
Created: 2025-05-21
Project: RoadInjuryAI
License: MIT
"""

import streamlit as st

from scripts.data_preprocessing import (load_data,
                               clean_data_pipeline,
                               data_encoding)
from scripts.model_training import random_forest, xg_boost
from scripts.vizualisation import class_repartion


def main():
    """
    Streamlit app to load, clean, and encode data, train models,
    and display class distribution and evaluation metrics.
    """
    st.title("🚀 Model Training and Evaluation Dashboard")

    st.subheader("1. Data Loading and Preprocessing")
    st.info("⏳ Loading and cleaning data...")
    df = load_data()
    df_clean = clean_data_pipeline(df)
    df_encoded, X_train_data = data_encoding(df_clean)

    st.success("✅ Data loaded and preprocessed.")

    # Display class distribution
    st.subheader("2. Class Distribution")
    class_repartion(df_encoded['grav'])

    # Train Random Forest
    st.subheader("3. Model Training: Random Forest")
    st.info("⏳ Training Random Forest model...")
    random_forest(X_train_data, df_encoded['grav'])

    # Train XGBoost
    st.subheader("4. Model Training: XGBoost")
    st.info("⏳ Training Gradient Boosting model...")
    xg_boost(X_train_data, df_encoded['grav'])


if __name__ == "__main__":
    main()