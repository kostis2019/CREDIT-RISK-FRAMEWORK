import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from . import settings

# load input file

def input_load(year_start, year_end, columns_to_use=None):

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
        chunk[settings.COLUMN_YEAR] = chunk[settings.COLUMN_DATE].dt.year
        # select years
        chunk = chunk[(chunk[settings.COLUMN_YEAR] >= year_start) & (chunk[settings.COLUMN_YEAR] <= year_end)]
        # append
        chunks.append(chunk)

    this_dataset = pd.concat(chunks, ignore_index=True)
    this_dataset = this_dataset[sorted(this_dataset.columns)]

    return this_dataset

# split train/test/oot datasets

def time_split(df, date_column, windows, target=settings.COLUMN_TARGET, features_excl=None, verbose=True,):

        if features_excl is None:
            features_excl = []

        splits = {}

        feature_cols = [
            c for c in df.columns
            if c not in features_excl + [target]
        ]

        for name, (start_year, end_year) in windows.items():

            mask = (
                (df[date_column] >= start_year) &
                (df[date_column] <= end_year)
            )

            df_split = df.loc[mask].copy()

            splits[name] = {
                "df": df_split,
                "X": df_split[feature_cols],
                "y": df_split[target],
                "years": (start_year, end_year),
                "n_rows": len(df_split),
                "default_rate": df_split[target].mean(),
            }

        if verbose:

            print("\n" + "=" * 60)
            print("SPLIT SUMMARY")
            print("=" * 60)

            for name, split in splits.items():

                print(f"\n{name.upper()}")
                print(f"Years        : {split['years'][0]} - {split['years'][1]}")
                print(f"Shape (X, y) : {split['X'].shape}, {split['y'].shape}")

                print("\nYear distribution")
                print(split["df"][date_column].value_counts().sort_index())

                print(f"\nDefault rate : {split['y'].mean():.4f}")
                print("-" * 60)

        return splits

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
