import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# load input file

def input_load(year_start, year_end, column_date, column_year, columns_to_use=None):

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
        chunk[column_year] = chunk[column_date].dt.year
        # select years
        chunk = chunk[(chunk[column_year] >= year_start) & (chunk[column_year] <= year_end)]
        # append
        chunks.append(chunk)

    this_dataset = pd.concat(chunks, ignore_index=True)
    this_dataset = this_dataset[sorted(this_dataset.columns)]

    return this_dataset

# split train/test/oot datasets

def time_split(df, date_column, windows, target, features_excl=None, verbose=True,):

        if features_excl is None:
            features_excl = []

        splits = {}

        feature_cols = [
            c for c in df.columns
            if c not in features_excl + [target] + [date_column]
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

                print(f"\nTarget mean : {split['y'].mean():.4f}")
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

# make homogeneous LGD-EAD portfolio

def make_homogeneous_portfolio(
    df,
    ead_col,
    lgd_col,
    pd_threshold,
    ead_result_col="EAD_hom",
    lgd_result_col="LGD_hom"
):
    """
    Create a homogeneous approved portfolio while preserving
    total EAD and baseline Expected Loss.

    Approval criterion: PD <= pd_threshold.
    Original PD heterogeneity is retained.
    """

    # Select approved portfolio
    result = df.loc[df["PD"] <= pd_threshold].copy()

    if result.empty:
        raise ValueError("No observations satisfy the PD threshold.")

    # Original approved portfolio characteristics
    n = len(result)
    total_ead = result[ead_col].sum()

    baseline_el = (
        result["PD"] *
        result[ead_col] *
        result[lgd_col]
    ).sum()

    # Homogeneous EAD
    ead_const = total_ead / n

    # Homogeneous LGD calibrated to preserve baseline EL
    pd_sum = result["PD"].sum()

    if ead_const <= 0 or pd_sum <= 0:
        raise ValueError("Cannot calibrate homogeneous LGD.")

    lgd_const = baseline_el / (ead_const * pd_sum)

    if not 0 <= lgd_const <= 1:
        raise ValueError(
            f"Calibrated LGD ({lgd_const:.4f}) "
            "is outside the valid range [0, 1]."
        )

    # Assign homogeneous values
    result[ead_result_col] = ead_const
    result[lgd_result_col] = lgd_const

    return result