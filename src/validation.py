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
