import pandas as pd
import numpy as np
from pathlib import Path
from IPython.display import Image, display
from playwright.async_api import async_playwright
from sklearn.utils import estimator_html_repr

# utility: save as .png

async def save_pipeline_diagram(
    estimator,
    output_path,
    width=1800,
    height=1200,
    display_image=True,
):
    """
    Save a scikit-learn estimator/pipeline diagram as a PNG.

    Parameters
    ----------
    estimator : sklearn estimator
        Trained estimator or pipeline.

    output_path : str or Path
        Output PNG path.

    width : int, default=1800
        Browser viewport width.

    height : int, default=1200
        Browser viewport height.

    display_image : bool, default=True
        Whether to display the generated image inside the notebook.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html_path = output_path.with_suffix(".html")

    # Create temporary HTML
    html_path.write_text(
        estimator_html_repr(estimator),
        encoding="utf-8",
    )

    # Render HTML with Playwright
    async with async_playwright() as p:

        browser = await p.chromium.launch()

        page = await browser.new_page(
            viewport={
                "width": width,
                "height": height,
            }
        )

        await page.goto(
            f"file://{html_path.resolve()}",
            wait_until="networkidle",
        )

        await page.screenshot(
            path=str(output_path),
            full_page=True,
        )

        await browser.close()

    # Remove temporary HTML
    html_path.unlink(missing_ok=True)

    # Display in notebook
    if display_image:
        display(Image(filename=str(output_path)))

    return output_path

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