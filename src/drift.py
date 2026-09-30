import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from . import settings

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def _calculate_numeric_drift(
    reference_values,
    comparison_values,
    n_bins=10,
):
    """
    Calculate KS statistic, p-value and PSI for a numeric variable.
    """

    ks_statistic, p_value = ks_2samp(
        reference_values,
        comparison_values,
    )

    # Create quantile-based bins from the reference population
    if len(np.unique(reference_values)) > 1:

        quantile_edges = np.quantile(
            reference_values,
            np.linspace(0, 1, n_bins + 1),
        )

        bin_edges = np.unique(quantile_edges)

        if len(bin_edges) > 2:

            reference_bins = pd.cut(
                reference_values,
                bins=bin_edges,
                include_lowest=True,
            )

            comparison_bins = pd.cut(
                comparison_values,
                bins=bin_edges,
                include_lowest=True,
            )

            reference_dist = (
                pd.Series(reference_bins)
                .value_counts(normalize=True, sort=False)
            )

            comparison_dist = (
                pd.Series(comparison_bins)
                .value_counts(normalize=True, sort=False)
            )

            # Ensure both distributions use exactly the same bins
            comparison_dist = comparison_dist.reindex(
                reference_dist.index,
                fill_value=0,
            )

            epsilon = 1e-6

            reference_dist = reference_dist.clip(
                lower=epsilon
            )

            comparison_dist = comparison_dist.clip(
                lower=epsilon
            )

            psi = (
                (comparison_dist - reference_dist)
                * np.log(
                    comparison_dist / reference_dist
                )
            ).sum()

        else:
            psi = 0.0

    else:
        psi = 0.0

    return ks_statistic, p_value, psi


def _calculate_categorical_drift(
    reference_values,
    comparison_values,
):
    """
    Calculate PSI for a categorical variable.

    KS statistic and p-value are not applicable.
    """

    reference_dist = (
        pd.Series(reference_values)
        .value_counts(normalize=True)
    )

    comparison_dist = (
        pd.Series(comparison_values)
        .value_counts(normalize=True)
    )

    # Use the union of categories observed in both periods
    categories = reference_dist.index.union(
        comparison_dist.index
    )

    reference_dist = reference_dist.reindex(
        categories,
        fill_value=0,
    )

    comparison_dist = comparison_dist.reindex(
        categories,
        fill_value=0,
    )

    epsilon = 1e-6

    reference_dist = reference_dist.clip(
        lower=epsilon
    )

    comparison_dist = comparison_dist.clip(
        lower=epsilon
    )

    psi = (
        (comparison_dist - reference_dist)
        * np.log(
            comparison_dist / reference_dist
        )
    ).sum()

    return psi


def calculate_drift(
    df,
    numeric_variables,
    categorical_variables,
    time_column,
    reference_period,
    comparison_periods,
    n_bins=10,
    threshold=0.05,
):
    """
    Calculate distributional drift between a reference period
    and one or more comparison periods.

    Numeric variables:
        KS statistic
        KS p-value
        PSI

    Categorical variables:
        PSI only

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    numeric_variables : list
        Numeric variables to monitor.

    categorical_variables : list
        Categorical variables to monitor.

    time_column : str
        Date column used to determine the observation year.

    reference_period : tuple
        Start and end year of the reference population.

    comparison_periods : list of tuples
        Periods to compare against the reference population.

    n_bins : int
        Number of quantile bins for numeric PSI.

    threshold : float
        Significance threshold for the KS test.
    """

    data = df.copy()

    data["_drift_year"] = data[time_column].dt.year

    reference_mask = (
        (data["_drift_year"] >= reference_period[0])
        & (data["_drift_year"] <= reference_period[1])
    )

    reference_data = data.loc[reference_mask]

    results = []

    variables = [
        (variable, "numeric")
        for variable in numeric_variables
    ] + [
        (variable, "categorical")
        for variable in categorical_variables
    ]

    for comparison_period in comparison_periods:

        comparison_mask = (
            (data["_drift_year"] >= comparison_period[0])
            & (data["_drift_year"] <= comparison_period[1])
        )

        comparison_data = data.loc[comparison_mask]

        for variable, variable_type in variables:

            reference_values = (
                reference_data[variable]
                .dropna()
                .to_numpy()
            )

            comparison_values = (
                comparison_data[variable]
                .dropna()
                .to_numpy()
            )

            if (
                len(reference_values) == 0
                or len(comparison_values) == 0
            ):
                continue

            if variable_type == "numeric":

                ks_statistic, p_value, psi = (
                    _calculate_numeric_drift(
                        reference_values,
                        comparison_values,
                        n_bins=n_bins,
                    )
                )

                ks_significant = p_value < threshold

            else:

                psi = _calculate_categorical_drift(
                    reference_values,
                    comparison_values,
                )

                ks_statistic = np.nan
                p_value = np.nan
                ks_significant = np.nan

            results.append({
                "Variable": variable,
                "Type": variable_type,
                "Reference": (
                    f"{reference_period[0]}"
                    f"-{reference_period[1]}"
                ),
                "Comparison": (
                    f"{comparison_period[0]}"
                    f"-{comparison_period[1]}"
                ),
                "KS": ks_statistic,
                "p_value": p_value,
                "KS_significant": ks_significant,
                "PSI": psi,
            })

    return pd.DataFrame(results)