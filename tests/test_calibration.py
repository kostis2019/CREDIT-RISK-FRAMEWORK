import numpy as np
import pandas as pd

from src.calibration import intercept_recalibration

def test_intercept_recalibration():

    df = pd.DataFrame({
        "PD": [0.1, 0.2, 0.3, 0.2],
        "y" : [0, 0, 1, 1],
    })

    pd_calibrated, delta = intercept_recalibration(df["y"], df["PD"])

    result          = delta
    result_expected = np.log(0.5 / (1 - 0.5)) - np.log(0.2 / (1 - 0.2))

    np.testing.assert_almost_equal(
        result,
        result_expected,
        decimal=10
    )