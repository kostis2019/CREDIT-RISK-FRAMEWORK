import numpy as np
import pandas as pd

from src.preprocessing import ReplaceMinusOne, time_split

def test_replaceminusone():

    df = pd.DataFrame({
        "column1": [0.1    ,   1  , '-1'  ],
        "column2": [0.1    ,  '1' ,  -1   ],
        "column3": ['-1'   ,  -2  ,  -1   ],
    })

    result = ReplaceMinusOne().fit_transform(df)

    expected = pd.DataFrame({
        "column1": [0.1    ,  1   , np.nan],
        "column2": [0.1    , '1'  , np.nan],
        "column3": [np.nan , -2   , np.nan],
    })

    pd.testing.assert_frame_equal(result, expected, check_dtype=False)

def test_time_split():

    df = pd.DataFrame({
        "a_date"  : [2017  ,2018  ,2019   ],
        "feature1": [0.1   ,0.2   ,0.3    ],
        "feature2": ['hot' ,'hot' ,'cold' ],
        "target"  : [0     ,0     ,1      ],
    })

    windows = {"TRAIN": (2017, 2018), "TEST":  (2019, 2019)}
    splits  = time_split(df, 'a_date', windows, "target")

    X_TRAIN = splits["TRAIN"]["X"]
    y_TRAIN = splits["TRAIN"]["y"]
    X_TEST  = splits["TEST"]["X"]
    y_TEST  = splits["TEST"]["y"]

    X_TRAIN_expected = pd.DataFrame({
        "a_date"  : [2017  ,2018   ],
        "feature1": [0.1   ,0.2    ],
        "feature2": ['hot' ,'hot'  ],
    })

    y_TRAIN_expected = pd.Series(
        [0, 0],
        name="target"
    )

    X_TEST_expected = pd.DataFrame({
        "a_date"  : [2019   ],
        "feature1": [0.3    ],
        "feature2": ['cold' ],
    })

    y_TEST_expected = pd.Series(
        [1],
        name="target"
    )      

    pd.testing.assert_frame_equal(X_TRAIN.reset_index(drop=True) , X_TRAIN_expected, check_dtype=False)
    pd.testing.assert_frame_equal(X_TEST.reset_index(drop=True)  , X_TEST_expected , check_dtype=False)

    pd.testing.assert_series_equal(y_TRAIN.reset_index(drop=True), y_TRAIN_expected, check_dtype=False)
    pd.testing.assert_series_equal(y_TEST.reset_index(drop=True) , y_TEST_expected , check_dtype=False)