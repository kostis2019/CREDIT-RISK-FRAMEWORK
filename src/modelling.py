import numpy as np
import pandas as pd

# function: apply pipeline 
#           - calculate PD
#           - calculate EL 
#           - sensitivity on LGD (optional)

def apply_pipe(df, pipeline, sensitivity_lgd=False):

    df = df.copy()
    df["PD"]    = pipeline.predict_proba(df)[:,1]
    df["EL"]    = df["PD"] * df["Amount"] * df["LossGivenDefault"]

    if sensitivity_lgd is True: 
        df["EL_10"] = df["PD"] * df["Amount"] * 0.10
        df["EL_15"] = df["PD"] * df["Amount"] * 0.15
        df["EL_25"] = df["PD"] * df["Amount"] * 0.25
        df["EL_90"] = df["PD"] * df["Amount"] * 0.90

    return df