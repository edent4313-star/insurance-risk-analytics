from operator import le

from pyparsing import col
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, mean_squared_error
import logging
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBClassifier, XGBRegressor


from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    roc_auc_score
)


def prepare_model_data(df):

    model_df = df.copy()

    categorical_cols = [
        "Gender",
        "Province",
        "VehicleType",
        "make"
    ]

    encoders = {}

    for col in categorical_cols:

        le = LabelEncoder()

        model_df[col] = le.fit_transform(
            model_df[col].astype(str)
        )

        encoders[col] = le

    features = [
        "CustomValueEstimate",
        "kilowatts",
        "Cylinders",
        "Province",
        "Gender",
        "VehicleType",
        "make"
    ]

    X = model_df[features]

    y_classification = (
        model_df["TotalClaims"] > 0
    ).astype(int)

    y_regression = model_df["TotalClaims"]

    return (
        X,
        y_classification,
        y_regression,
        features
    )


def train_claim_classifier(X, y):

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return (
        model,
        X_test,
        y_test,
        predictions,
        accuracy
    )


def train_claim_regressor(X, y):

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    return (
        model,
        X_test,
        y_test,
        predictions,
        mae
    )

def train_severity_model(X, y_reg):
    """Trains model to predict HOW MUCH (only for rows where claims > 0)."""
    # Filter for only positive claims
    mask = y_reg > 0
    X_sub = X[mask]
    y_sub = y_reg[mask]

    if len(y_sub) < 10:
        return None, "Not enough claim data to train severity model.", None, None, None, None

    X_train, X_test, y_train, y_test = train_test_split(X_sub, y_sub, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    return (model, X_test, y_test, preds, rmse, r2)





def train_severity_models(X_train, y_train):
    """Fits Linear Regression, Random Forest, and XGBoost for severity estimation."""
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ),
        "XGBoost Regressor": XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
        ),
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def train_frequency_models(X_train, y_train):
    """Fits Logistic Regression, Random Forest, and XGBoost for probability of claims."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ),
        "XGBoost Classifier": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
        ),
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models