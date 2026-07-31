import numpy as np
import pandas as pd
from src.feature_engineering import create_target_def12
from src.preprocessing import my_input_load
from src.utils import display_table

# function: apply pipeline 
#           - calculate PD
#           - calculate EL 
#           - sensitivity on LGD (optional)

def apply_pipe(df, pipeline, calculate_el=False, sensitivity_lgd=False):

    df = df.copy()
    df["PD"]    = pipeline.predict_proba(df)[:,1]

    if calculate_el is True:
        # calculate expected loss using realised LGD 
        df["EL"]    = df["PD"] * df["Amount"] * df["LossGivenDefault"]

        if sensitivity_lgd is True:
            # calculate expected loss using LGD 10%, 15%, 25%, 90% 
            df["EL_10"] = df["PD"] * df["Amount"] * 0.10
            df["EL_15"] = df["PD"] * df["Amount"] * 0.15
            df["EL_25"] = df["PD"] * df["Amount"] * 0.25
            df["EL_90"] = df["PD"] * df["Amount"] * 0.90

    return df

# function: estimate LGD

def estimate_lgd(df, yr_start, yr_end, method, one_value=0.15):

    df = df.copy()

    lgd_dataset = my_input_load(yr_start, yr_end)
    lgd_dataset, dr_summary = create_target_def12(lgd_dataset)
    lgd_dataset = lgd_dataset.loc[lgd_dataset["default12"] == 1]

    ###### ESTIMATE ######

    print('LGD estimation method: ', method)

    if method == 'one_value':

        value    = one_value

    if method == 'historical_avg':

        value    = lgd_dataset['LossGivenDefault'].mean()

    if method == 'exposure_bands':
        
        bands       = [0, 1000, 2500, 5000, 7500, 10000, 20000, np.inf]
        band_labels = ["0–1k", "1–2.5k", "2.5–5k", "5–7.5k", "7.5–10k", "10–20k", ">20k"]

        df_eval = pd.DataFrame({
        "amount": lgd_dataset['Amount'], 
        "lgd"   : lgd_dataset['LossGivenDefault'] 
        })

        df_eval["amount_bin"] = pd.cut(df_eval["amount"], bins=bands, labels=band_labels, include_lowest=True)
        
        lgd_table = (
        df_eval.groupby("amount_bin", observed=False)
        .agg(
            counts=("amount", "count"),
            amount=("amount", "sum"),
            lgd   =("lgd", "mean"),
        ))
        display(lgd_table)

        lgd_mapping = lgd_table["lgd"].to_dict()

    if method == 'linear_regression':
        
        print('in progress')

    if method == 'gradient_boosting_regression':
        
        print('in progress')

    ###### APPLY ######

    if method == 'one_value':

        df["LGD"]    = value

    if method == 'historical_avg':

        df["LGD"]    = value

    if method == "exposure_bands":

        df["amount_bin"] = pd.cut(
            df["Amount"],
            bins=bands,
            labels=band_labels,
            include_lowest=True
        )

        df["LGD"] = df["amount_bin"].map(lgd_mapping)
        df.drop(columns="amount_bin", inplace=True)

    return df