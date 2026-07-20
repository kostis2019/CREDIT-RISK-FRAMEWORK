import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from IPython.display import display

# load input file

def my_input_load(year_start, year_end, columns_to_use=None):

    file_path = "../data/raw/LoanData_(DS_Home_Task).csv"

    converters = {
        "LiabilitiesTotal": lambda x: float(x) if str(x).replace('.', '', 1).isdigit() else np.nan,
        "MonthlyPaymentDay": lambda x: int(x) if str(x).isdigit() else pd.NA
    }

    chunks = []

    for chunk in pd.read_csv(  # pyright: ignore[reportCallIssue]
        file_path,
        sep=";",
        usecols=columns_to_use,
        converters=converters,
        parse_dates=["LoanApplicationStartedDate", 
                    "LoanDate", 
                    "FirstPaymentDate",
                    "DefaultDate"],
        chunksize=100_000,
        low_memory=False
    ):
        # select country
        chunk = chunk[chunk["Country"] == "EE"]
        # create LoanYear
        chunk["LoanYear"] = chunk["LoanDate"].dt.year
        # select years
        chunk = chunk[(chunk["LoanYear"] >= year_start) & (chunk["LoanYear"] <= year_end)]
        # append
        chunks.append(chunk)

    this_dataset = pd.concat(chunks, ignore_index=True)
    this_dataset = this_dataset[sorted(this_dataset.columns)]

    return this_dataset

# split train/test/oot datasets

def my_time_split(full_dataset, train_yr_start, train_yr_end, test_yr_start, test_yr_end):

    # train/test window
    trai = full_dataset[(full_dataset["LoanYear"] >= train_yr_start) & (full_dataset["LoanYear"] <= train_yr_end)]
    test = full_dataset[(full_dataset["LoanYear"] >= test_yr_start)  & (full_dataset["LoanYear"] <= test_yr_end)]

    # features to NOT use
    features_excl = ["default12","LoanDate","DefaultDate","LoanYear","Country",
                        "CreditScoreEsMicroL","NrOfDependants",
                        "LoanApplicationStartedDate","FirstPaymentDate"]

    # features to use
    all_features = [col for col in trai.columns if col not in features_excl]

    # split
    X_trai = trai[all_features]
    y_trai = trai["default12"]
    X_test = test[all_features]
    y_test = test["default12"] 

    # check shapes
    print("Train size:", X_trai.shape, y_trai.shape)
    print("Test size :", X_test.shape, y_test.shape)

    # check counts
    print(trai["LoanYear"].value_counts())
    print(test["LoanYear"].value_counts())

    # check DR
    print("Train default rate:", y_trai.mean())
    print("Test default rate :", y_test.mean())

    return X_trai, y_trai, X_test, y_test

# transformer: replace -1s

class ReplaceMinusOne(BaseEstimator, TransformerMixin):

    def __init__(self, verbose=False):
        self.verbose = verbose

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if self.verbose:
            for col in X.columns:

                n_numeric = (X[col] == -1).sum()
                n_string  = (X[col] == "-1").sum()
                n_total   = n_numeric + n_string

                if n_total > 0:
                    print(f"{col}: replacing {n_total:,} values of -1 with NaN")

        X = X.replace(-1, np.nan)
        X = X.replace("-1", np.nan)

        return X
