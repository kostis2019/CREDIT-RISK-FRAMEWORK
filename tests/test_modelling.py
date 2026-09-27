import pandas as pd

from src.modelling import estimate_el

def test_estimate_el():
    df = pd.DataFrame({
        "PD": [0.10, 0.20, 0.05],
        "LGD": [0.40, 0.50, 0.30],
        "Amount": [1000, 2000, 10000],
    })

    result = estimate_el(df)

    expected = [40.0, 200.0, 150.0]

    assert result["EL"].tolist() == expected