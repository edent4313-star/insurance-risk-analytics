from src.data_loader import load_data


def test_load_data():

    df = load_data(
        "data/insurance_data.csv"
    )

    assert len(df) > 0