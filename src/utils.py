import pandas as pd
import numpy as np
from pathlib import Path
from IPython.display import display
import matplotlib.pyplot as plt

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
