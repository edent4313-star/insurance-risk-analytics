import pandas as pd
import pytest
from src.preprocessing import clean_insurance_data,clean_and_engineer_data, prepare_and_split_data
from src.preprocessing import (
    calculate_loss_ratio
)


def test_loss_ratio():
    """Tests the loss ratio calculation with standard and edge-case inputs."""
    
    # 1. Standard Case
    try:
        df = pd.DataFrame({
            "TotalClaims": [100.0],
            "TotalPremium": [200.0]
        })
        
        df = calculate_loss_ratio(df)
        
        # Check if column exists before asserting
        assert "LossRatio" in df.columns, "LossRatio column not created."
        assert df["LossRatio"][0] == 0.5, f"Expected 0.5, got {df['LossRatio'][0]}"
        
    except Exception as e:
        print(f"Test failed in standard case: {e}")
        raise

    # 2. Edge Case: Division by Zero
    try:
        df_zero = pd.DataFrame({
            "TotalClaims": [100.0],
            "TotalPremium": [0.0]
        })
        
        # This should handle the division by zero gracefully 
        # (usually by returning 0 or np.nan depending on your function logic)
        df_zero = calculate_loss_ratio(df_zero)
        
        # Adjust expectation based on how you want your function to behave
        assert np.isnan(df_zero["LossRatio"][0]) or df_zero["LossRatio"][0] == 0.0
        print("Edge case (Zero Premium) passed.")
        
    except ZeroDivisionError:
        print("Caught expected ZeroDivisionError.")
    except Exception as e:
        print(f"Unexpected error in edge case: {e}")
        raise




def test_clean_insurance_data():
    data = {
        'TotalPremium': [100, 105, 110, 100000], # 100k is an outlier
        'TotalClaims':[0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    cleaned = clean_insurance_data(df)
    
    # Assert outlier was removed
    assert 100000 not in cleaned['TotalPremium'].values
    assert len(cleaned) == 3




def test_leak_free_pipeline():
    mock_df = pd.DataFrame(
        {
            "TransactionMonth": ["2026-01-01", "2026-02-01", "2026-03-01"],
            "RegistrationYear": [2020, 2018, 2015],
            "TotalClaims": [0, 4500.00, 0],
            "Gender": ["Male", "Female", "Male"],
        }
    )

    cleaned = clean_and_engineer_data(mock_df)
    X_train, X_test, y_train, y_test = prepare_and_split_data(
        cleaned, task_type="severity_regression"
    )

    # Validates that severity isolation tracks claims > 0 perfectly
    assert (len(X_train) + len(X_test)) == 1
    assert "VehicleAge" in X_train.columns