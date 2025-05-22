"""
model_training.py

This module contains training routines for classification models
used to predict road accident severity. It currently supports:

- Random Forest
- Gradient Boosting (XGBoost-style)

Each function handles its own data splitting, training, and evaluation.

Author: Anthony Morin
Created: 2025-05-21
Project: RoadInjuryAI
License: MIT
"""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, recall_score, f1_score
from tabulate import tabulate
from colorama import Fore, Style, init

def random_forest(X, y):
    """
    Train and evaluate a Random Forest classifier.

    Steps:
    - Normalize feature data
    - Split into training and test sets
    - Train RandomForestClassifier
    - Evaluate accuracy on both train and test sets

    Args:
        X (pd.DataFrame or np.ndarray): Feature matrix.
        y (pd.Series or np.ndarray): Target labels.
    """
    X_normalized = normalize(X.values)
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

    model_rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    model_rf.fit(X_train, y_train)

    predictions_train = model_rf.predict(X_train)
    predictions_test = model_rf.predict(X_test)

    display_metrics("Random Forest", y_train, predictions_train, y_test, predictions_test)


def xg_boost(X, y):
    """
    Train and evaluate a Gradient Boosting classifier.

    Steps:
    - Split data into training and test sets
    - Train GradientBoostingClassifier with custom hyperparameters
    - Evaluate accuracy on both train and test sets

    Args:
        X (pd.DataFrame or np.ndarray): Feature matrix.
        y (pd.Series or np.ndarray): Target labels.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_boosting = GradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.2,
        max_depth=5,
        max_features="sqrt",
        subsample=0.95,
        n_estimators=200,
        random_state=42
    )

    model_boosting.fit(X_train, y_train)

    predictions_train = model_boosting.predict(X_train)
    predictions_test = model_boosting.predict(X_test)

    display_metrics("XGBoost-style Gradient Boosting", y_train, predictions_train, y_test, predictions_test)


init(autoreset=True)  # reset colors after each print

def display_metrics(model_name, y_train, pred_train, y_test, pred_test):
    """
    Display formatted evaluation metrics in the terminal with colors and table layout.

    Args:
        model_name (str): Name of the model.
        y_train, y_test (array): True labels.
        pred_train, pred_test (array): Predicted labels.
    """
    train_acc = accuracy_score(y_train, pred_train)
    test_acc = accuracy_score(y_test, pred_test)
    recall = recall_score(y_test, pred_test, average='macro')
    f1 = f1_score(y_test, pred_test, average='macro')

    print(f"\n{Fore.CYAN}{Style.BRIGHT}📊 Model Evaluation – {model_name}\n")

    table = [
        ["Train Accuracy", f"{train_acc:.2%}"],
        ["Test Accuracy", f"{test_acc:.2%}"],
        ["Recall (macro)", f"{recall:.3f}"],
        ["F1-Score", f"{f1:.3f}"]
    ]

    print(tabulate(table, headers=["Metric", "Value"], tablefmt="fancy_grid"))