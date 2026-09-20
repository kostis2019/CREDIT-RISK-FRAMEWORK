import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from IPython.display import display

# create target variable

def create_target_def12(full_dataset):

    # first and last loan date
    first_date = full_dataset.LoanDate.min()
    last_date  = full_dataset.LoanDate.max()

    # first and last default date
    first_defdate = full_dataset.DefaultDate.min()
    last_defdate  = full_dataset.DefaultDate.max()

    # print
    print('LoanDate range:    ', first_date, last_date)
    print('+12 months:        ', first_date + pd.DateOffset(months=12), last_date + pd.DateOffset(months=12))
    print('DefaultDate range: ', first_defdate, last_defdate)

    # default condition
    full_dataset["default12"] = np.where(
        # define a 12-month window from LoanDate
        # check if DefaultDate is in
        (full_dataset["DefaultDate"].notna()) & 
        (full_dataset["DefaultDate"] <= full_dataset["LoanDate"] + pd.DateOffset(months=12)),
        1,  # default
        0   # no default within 12 months
    )

    # display DR by year
    # create LoanYear
    full_dataset["LoanYear"] = full_dataset["LoanDate"].dt.year

    # group by year and aggregate
    DR_summary = full_dataset.groupby("LoanYear").agg(
        num_loans=("LoanDate", "count"),
        num_defaults=("default12", "sum")
    ).reset_index()

    # calculate default rate
    DR_summary["default_rate"] = DR_summary["num_defaults"] / DR_summary["num_loans"]

    # display the table
    display(DR_summary.style.hide(axis="index"))

    return full_dataset, DR_summary

# handle missing values (conventionally)

def handle_missing_values(X_trai, X_test, do_not_handle=[]):

    # MISSING VALUES

    # separate nums/cats
    cat_cols = X_trai.select_dtypes(include=["category"]).columns
    num_cols = X_trai.select_dtypes(include=["int64", "float64"]).columns

    # remove columns that should not be handled
    cat_cols = cat_cols.difference(do_not_handle)
    num_cols = num_cols.difference(do_not_handle)

    # substitution values for numerical variables: median in training set
    median_vals = X_trai[num_cols].median()

    print('SUBSTITUTION VALUES FOR NUMS: ')
    print(median_vals)

    # substitution values for categorical variables: most common in training set
    common_vals = X_trai[cat_cols].mode().iloc[0]

    print('SUBSTITUTION VALUES FOR CATS: ')
    print(common_vals)

    # replace NaNs in numerical variables
    X_trai[num_cols] = X_trai[num_cols].fillna(median_vals)
    X_test[num_cols] = X_test[num_cols].fillna(median_vals)

    # replace NaNs in categorical variables
    X_trai[cat_cols] = X_trai[cat_cols].fillna(common_vals)
    X_test[cat_cols] = X_test[cat_cols].fillna(common_vals)

    return X_trai, X_test

# map special features

class SpecialMappings(BaseEstimator, TransformerMixin):

    def __init__(self, verbose=False):

        self.verbose = verbose

    # define all mappings here
    
        self.mappings = {
            
            "EmploymentDurationCurrentEmployer": {
                "TrialPeriod": 0,
                "UpTo1Year": 1,
                "UpTo2Years": 2,
                "UpTo3Years": 3,
                "UpTo4Years": 4,
                "UpTo5Years": 5,
                "MoreThan5Years": 6,
                "Retiree": 7,
                "Other": np.nan
            },

            "WorkExperience": {
                "LessThan2Years": 0,
                "2To5Years": 2,
                "5To10Years": 5,
                "10To15Years": 10,
                "15To25Years": 15,
                "MoreThan25Years": 25,
                "Other": np.nan
            },

            # "NewCreditCustomer": {
            #     "True": 1,
            #     "False": 0,
            #     "Other": np.nan
            # }

            # 👉 add more variables here
            # "AnotherVariable": {
            #     "A": 1,
            #     "B": 2,
            #     "C": np.nan
            # }
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        for col, mapping in self.mappings.items():
            if col in X.columns:

                if self.verbose:
                    print("- Before mapping: ")
                    print(X[col].value_counts(dropna=False))

                X[col] = X[col].map(mapping)
                X[col] = X[col].astype("Int64")

                if self.verbose:
                    print("- After mapping: ")
                    print(X[col].value_counts(dropna=False))

        return X

# map special features: mortgage dataset

class SpecialMappingsMortgage(BaseEstimator, TransformerMixin):

    def __init__(self, verbose=False):

        self.verbose = verbose

    # define all mappings here
    
        self.mappings = {
            
            # "PropertyType": {
            #     "Apartment": 1,
            #     "Single-family house": 2,
            #     "Townhouse": 3,
            #     "Double-family house": 4,
            #     "Residential real estate": 5,
            #     "Apartment block": 6,
            #     "Unknown": np.nan
            # },

            # 👉 add more variables here
            # "AnotherVariable": {
            #     "A": 1,
            #     "B": 2,
            #     "C": np.nan
            # }
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        for col, mapping in self.mappings.items():
            if col in X.columns:

                if self.verbose:
                    print("- Before mapping: ")
                    print(X[col].value_counts(dropna=False))
        
                # mapping
                X[col] = X[col].map(mapping)

                # ordinal categorical variables
                X[col] = X[col].astype("Int64")

                # nominal categorical variables
                if col == "PropertyType": 
                    X[col] = X[col].astype("category")

                if self.verbose:
                    print("- After mapping: ")
                    print(X[col].value_counts(dropna=False))

        return X

# select features (selects pre-defined selected features)

class FeatureSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, selected_features):
        self.selected_features = selected_features
    
    def fit(self, X, y=None):
        # infer feature names automatically
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns
        else:
            # fallback if numpy array
            self.feature_names = [f"f{i}" for i in range(X.shape[1])]
        
        missing = [f for f in self.selected_features if f not in self.feature_names]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        
        return self
    
    def transform(self, X):

        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns
        else:
            self.feature_names = pd.Index([f"f{i}" for i in range(X.shape[1])])
        
        return X[self.selected_features]