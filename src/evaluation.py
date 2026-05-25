from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)


def evaluate_regression(models: dict, X_test, y_test):
    """Returns a comparative dataframe of RMSE and R² metrics."""
    return {
        name: {
            "RMSE": root_mean_squared_error(y_test, model.predict(X_test)),
            "R2_Score": r2_score(y_test, model.predict(X_test)),
        }
        for name, model in models.items()
    }


def evaluate_classification(models: dict, X_test, y_test):
    """Returns a comparative dataframe of classification metrics."""
    res = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        res[name] = {
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, zero_division=0),
            "Recall": recall_score(y_test, preds, zero_division=0),
            "F1_Score": f1_score(y_test, preds, zero_division=0),
        }
    return res