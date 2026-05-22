from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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