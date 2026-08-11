import pandas as pd

# function: missing summary

def missing_summary(df):
    total_rows = len(df)
    
    summary = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percent": (df.isna().sum() / total_rows) * 100
    })
    
    #summary = summary[summary["missing_count"] > 0]  # keep only columns with missing
    summary = summary.sort_values(by="missing_percent", ascending=False)
    
    return summary

# function: validate model output

def validate_predictions(df):

    print("-" * 40)
    print("MODEL OUTPUT VALIDATION")
    print("-" * 40)

    summary = pd.DataFrame({
        "PD": [
            df["PD"].min(),
            df["PD"].mean(),
            df["PD"].max(),
            df["PD"].isna().sum()
        ],
        "LGD": [
            df["LGD"].min(),
            df["LGD"].mean(),
            df["LGD"].max(),
            df["LGD"].isna().sum()
        ]
    },
    index=["Min", "Mean", "Max", "NaNs"])

    display(summary.round(3))