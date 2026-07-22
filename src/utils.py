from pathlib import Path
from IPython.display import Image, display
from playwright.async_api import async_playwright
from sklearn.utils import estimator_html_repr


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