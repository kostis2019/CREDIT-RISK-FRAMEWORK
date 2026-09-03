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

    required = ["PD", "LGD"]   
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

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

# function: best model selector

def model_selector(experiment_metrics, metric="CORR"):
    """
    Select the best model based on validation performance.

    Parameters
    ----------
    experiment_metrics : pd.DataFrame
        Output table containing TRAIN and TEST metrics.
    metric : str
        Metric to use (default: CORR).

    Returns
    -------
    selection_table : pd.DataFrame
    best_model : pd.DataFrame
    """

    # reshape: one row per experiment
    selection_table = (
        experiment_metrics
        .pivot(
            index=["Experiment", "Method", "Variable"],
            columns="Dataset",
            values=metric,
        )
        .reset_index()
        .rename(columns={
            "TRAIN": f"Train {metric}",
            "TEST": f"Test {metric}",
        })
    )

    # remove preprocessing prefixes
    selection_table["Variable"] = (selection_table["Variable"].str.replace("num__", "", regex=False).str.replace("cat__", "", regex=False))

    # generalization gap
    selection_table["Generalization Gap"] = (
        selection_table[f"Train {metric}"] -
        selection_table[f"Test {metric}"]
    ).abs()

    # select best model:
    # highest validation metric, then smallest gap
    best_model = (
        selection_table
        .sort_values(
            by=[f"Test {metric}", "Generalization Gap"],
            ascending=[False, True]
        )
        .head(1)
    )

    return selection_table, best_model