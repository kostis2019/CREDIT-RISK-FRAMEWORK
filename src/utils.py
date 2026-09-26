import pandas as pd
import numpy as np
from IPython.display import display

# utility: clean table format

def clean_table_format(df, decimals=3):
    df = df.copy()

    def fmt(x):

        # Interval objects
        if isinstance(x, pd.Interval):

            left = x.left
            right = x.right

            # Replace tiny negative lower bound with 0
            if np.isclose(left, 0, atol=10**(-decimals)):
                left = 0

            left = f"{left:.{decimals}f}".rstrip("0").rstrip(".")
            right = f"{right:.{decimals}f}".rstrip("0").rstrip(".")

            return f"({left}, {right}]"

        # Numeric values
        if isinstance(x, (int, float, np.integer, np.floating)):
            # Display whole numbers as integers
            if float(x).is_integer():
                return str(int(x))

            # Otherwise remove trailing zeros
            return f"{float(x):.{decimals}f}"#.rstrip("0").rstrip(".")

        return x

    return df.map(fmt)

# utility: display table

def display_table(df, decimals=3, hide_index=True):
    df = clean_table_format(df, decimals)

    styler = df.style

    if hide_index:
        styler = styler.hide(axis="index")

    display(styler)

# utility: format policy table

def format_policy_table(df):

    formatted = df.copy()

    # 3 decimal places
    for metric in ["AUC", "KS", "Brier"]:
        if metric in formatted.index:
            formatted.loc[metric] = formatted.loc[metric].map(
                lambda x: f"{x:.3f}"
            )

    # Percentages
    for metric in ["Observed_DR", "Mean_PD", "EL_Rate"]:
        if metric in formatted.index:
            formatted.loc[metric] = formatted.loc[metric].map(
                lambda x: f"{x:.2%}"
            )

    # Whole numbers with thousands separator
    for metric in [
        "Exposure",
        "EL_Total",
        "Monte-Carlo Expected Loss",
        "Monte-Carlo Economic Capital"
    ]:
        if metric in formatted.index:
            formatted.loc[metric] = formatted.loc[metric].map(
                lambda x: f"{x:,.0f}"
            )

    return formatted

# utility: apply common plot style 

def apply_slide_style(ax, ax_top=None):

    ax.tick_params(axis="both", labelsize=11)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.title.set_size(14)
    ax.title.set_weight("bold")

    if ax_top is not None:
        ax_top.tick_params(axis="x", labelsize=11)
        ax_top.xaxis.label.set_size(12)
