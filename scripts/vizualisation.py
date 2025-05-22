"""
visualization.py

Visualization functions.

Author: Anthony Morin
Created: 2025-05-21
Project: RoadInjuryAI
License: MIT
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, f1_score


def class_repartion(y):
    """
    Display a pie chart showing the class distribution using Streamlit.

    Parameters:
    -----------
    y : array-like
        The target labels (classes).
    """
    values = np.unique(y, return_counts=True)[1]
    # Class labels:
    # 1 : Indemne
    # 2 : Tué
    # 3 : Hospitalisé
    # 4 : Blessé léger
    labels = ['Indemne', 'Tué', 'Hospitalisé', 'Blessé léger']
    colors = ['#fff100', '#e81123', '#ff8c00', '#ec008c']

    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    centre_circle = plt.Circle((0, 0), 0.70, color='white') # type: ignore
    ax.add_artist(centre_circle)
    ax.axis('equal')
    plt.tight_layout()

    st.pyplot(fig)


def display_metrics(model_name, y_train, pred_train, y_test, pred_test):
    """
    Display evaluation metrics for a classification model using Streamlit.

    Parameters:
    -----------
    model_name : str
        Name of the trained model.
    y_train, y_test : array-like
        True labels for training and test sets.
    pred_train, pred_test : array-like
        Predicted labels for training and test sets.
    """
    train_acc = accuracy_score(y_train, pred_train)
    test_acc = accuracy_score(y_test, pred_test)
    recall = recall_score(y_test, pred_test, average='macro')
    f1 = f1_score(y_test, pred_test, average='macro')

    st.markdown(f"### 📊 Evaluation Metrics – {model_name}")

    metrics_df = pd.DataFrame({
        "Metric": ["Train Accuracy", "Test Accuracy", "Recall (macro)", "F1-Score"],
        "Value": [f"{train_acc:.2%}", f"{test_acc:.2%}", f"{recall:.3f}", f"{f1:.3f}"]
    })

    st.table(metrics_df)