import pandas as pd

from src.feature_engineering import FeatureSelector

def test_featureselector():

    df = pd.DataFrame({
        "a_date":   [2017, 2018, 2019],
        "feature1": [0.1, 0.2, 0.3],
        "feature2": ["hot", "hot", "cold"],
        "feature3": [0, 0, 1],
    })

    selected_features = ["feature1", "feature2"]

    selector = FeatureSelector(selected_features)

    result = selector.fit_transform(df)

    result_expected = pd.DataFrame({
        "feature1": [0.1, 0.2, 0.3],
        "feature2": ["hot", "hot", "cold"],
    })

    pd.testing.assert_frame_equal(
        result,
        result_expected,
        check_dtype=True
    )