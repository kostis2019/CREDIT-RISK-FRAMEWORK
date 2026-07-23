import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from IPython.display import display
from src.utils import display_table

# plot: default rate per year

def my_dr_plot(DR_summary, year_start, year_end):

    # ==============================
    # PLOT DR PER YEAR
    # ==============================

    DR_summary = DR_summary[(DR_summary.LoanYear >= year_start) & (DR_summary.LoanYear <= year_end)]

    fig, ax1 = plt.subplots(figsize=(6,3))

    # Bars: number of loans
    ax1.bar(
        DR_summary["LoanYear"],
        DR_summary["num_loans"],
        color="lightskyblue",
        alpha=0.3
    )

    ax1.set_xlabel("Loan Origination Year")
    ax1.set_ylabel("Number of Loans")
    ax1.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)

    # Line: default rate
    ax2 = ax1.twinx()

    ax2.plot(
        DR_summary["LoanYear"],
        DR_summary["default_rate"],
        marker="o",
        markersize=8,
        markerfacecolor="limegreen",
        markeredgecolor="limegreen",
        color="limegreen",
        linewidth=2
    )

    ax2.set_ylabel("Default Rate")
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # annotate default rate
    for x, y in zip(DR_summary["LoanYear"], DR_summary["default_rate"]):
        ax2.text(x+0.3, y, f"{y:.1%}", ha="center", fontsize=11)

    plt.tight_layout()
    plt.show()

    return

# plot: variable vs default rate

def variable_vs_dr(variable, x_data, y_data):

    # ==============================
    # PLOT VARIABLE VS DEFAULT RATE
    # ==============================

    # -------- PLOT INPUTS ----------
    input_variable = variable
    df_plot = x_data.copy()
    df_plot["default12"] = y_data

    # ------- USER SETTINGS ---------
    binning_method = "equal"  # options: "quantile", "equal", "none"
    n_bins = 10

    # inspect values
    print(df_plot[input_variable].describe())

    # -------- BINNING OPTIONS --------------

    if binning_method == "quantile":
        df_plot["bin"] = pd.qcut(
            df_plot[input_variable],
            n_bins,
            duplicates="drop"
        )

    elif binning_method == "equal":
        df_plot["bin"] = pd.cut(
            df_plot[input_variable],
            n_bins
        )

    elif binning_method == "none":
        # no binning → use raw values
        summary = df_plot[[input_variable, "default12"]].copy()
        summary.columns = ["mean_value", "default_rate"]

    else:
        raise ValueError("binning_method must be 'quantile', 'equal', or 'none'")

    # -------- SUMMARIZE (if binned) --------

    if binning_method != "none":
        summary = df_plot.groupby("bin").agg(
            mean_value=(input_variable, "mean"),
            default_rate=("default12", "mean")
        ).reset_index()

    print(summary)

    # -------- PLOT --------

    plt.figure(figsize=(5,4))

    plt.plot(
        summary["mean_value"],
        summary["default_rate"],
        marker="o",
        markersize=8,
        markerfacecolor="black",
        markeredgecolor="black",
        color="black",
        linewidth=1.8
    )

    plt.xlabel(input_variable)
    plt.ylabel("Default Rate")
    plt.title(f"Default Rate vs {input_variable}")

    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    plt.tight_layout()
    plt.show()

# plot: calibration table and plot

def my_calibration(y_true, pd_pred, dataset_name="", show_cal_table=True, plot_title="Calibration Plot"):

    # 1. risk table

    # predictions and observations 
    probab = pd_pred # predicted PDs
    target = y_true  # 0/1

    # define bins 
    # a. equal bins:
    #bins = np.linspace(0, 1, 21)
    # b. manual bins:
    bins = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    df_eval = pd.DataFrame({
        "probability": probab, # predicted PDs
        "default":     target  # 0/1
    })

    df_eval["prob_bin"] = pd.cut(df_eval["probability"], bins=bins)

    table = (
        df_eval.groupby("prob_bin", observed=False)
        .agg(
            Count=("default", "count"),
            Defaults=("default", "sum"),
            Avg_PD=("probability", "mean")
        )
    )

    # Observed default rate
    table["Observed_Default_Rate"] = table["Defaults"] / table["Count"]

    # Confidence interval (95%)
    table["CI_std"] = np.sqrt(table["Observed_Default_Rate"] * (1 - table["Observed_Default_Rate"]) / table["Count"])

    # high and low
    table["CI_lower"] = table["Observed_Default_Rate"] - 1.96 * table["CI_std"]
    table["CI_upper"] = table["Observed_Default_Rate"] + 1.96 * table["CI_std"]

    # clean up (avoid negatives or >1)
    table["CI_lower"] = table["CI_lower"].clip(lower=0)
    table["CI_upper"] = table["CI_upper"].clip(upper=1)

    # Share of portfolio (BY NUMBER OF LOANS)
    table["Share_of_Portfolio"] = table["Count"] / table["Count"].sum()

    # show table
    table = table.reset_index()
    if (show_cal_table is True):
        display_table(table)

    # save table:
    # table.to_csv("calibration_table.csv", index=False)

    print("-" * 40)

    # 2. plot

    # Bin centers from interval
    bin_centers = [interval.mid for interval in table["prob_bin"]]

    fig, ax = plt.subplots(figsize=(5,4)) # i do this to: 1. show figure 2. return figure

    # Calibration lines:
    ax.plot(bin_centers, table["Avg_PD"],                color='lightskyblue', marker='o', label="Average Predicted PD")
    ax.plot(bin_centers, table["Observed_Default_Rate"], color='limegreen',    marker='o', label="Observed Default Rate")
    ax.plot([0,1],[0,1],'--',                            color='black'                   , label="Perfect Calibration")

    # Confidence interval shading:
    ax.fill_between(
        bin_centers,
        table["CI_lower"],
        table["CI_upper"],
        color='limegreen',
        alpha=0.2,
        label="95% CI (Observed DR)"
    )

    ax.set_xlabel("PD Bin")
    ax.set_ylabel("PD,DR")
    ax.set_title(plot_title)
    ax.legend()
    # -> set ticks to bin edges
    ax.set_xticks(bins)
    # -> add vertical grid aligned with bins
    ax.grid(axis='x', linestyle='-', alpha=0.5)
    # -> add horizontal grid 
    ax.grid(axis='y', linestyle='-', alpha=0.5)
    # optional: rotate labels if crowded
    ax.set_xticklabels(bins, rotation=45)
    
    plt.tight_layout()
    
    # show plot now!
    display(fig)
    plt.close(fig)

    print("-" * 40)

    return

# plot: default capture curve

def default_capture_curve(y_true, pd_pred):

    # Create DataFrame
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": pd_pred
    })

    # Sort by predicted PD (highest risk first)
    df = df.sort_values(by="y_pred", ascending=False).reset_index(drop=True)

    # Cumulative population %
    df["cum_population"] = np.arange(1, len(df)+1) / len(df)

    # Cumulative defaults captured
    df["cum_defaults"] = df["y_true"].cumsum()

    # Total defaults
    total_defaults = df["y_true"].sum()

    # Avoid division by zero
    df["cum_default_rate"] = df["cum_defaults"] / total_defaults if total_defaults > 0 else 0

    # 📊 Plot

    plt.figure(figsize=(5,4))

    plt.plot(df["cum_population"], df["cum_default_rate"], label="Logistic Regression", color="limegreen")

    # Random baseline (diagonal)
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random", color="lightskyblue")

    plt.xlabel("Cumulative % of Population")
    plt.ylabel("Cumulative % of Defaults Captured")
    plt.title("Default Capture Curve")
    plt.legend()
    plt.grid(True)

    plt.show()

    # print captured defaults
    # for i in range(df.shape[0]):
    #    print(f"cum_population={df['cum_population'].iloc[i]:5.5f}", f"cum_default_rate={df['cum_default_rate'].iloc[i]:5.5f}")

# plot: EL concentration / PDbins

def my_el_plot(table_EL, table_EL_x):

    # (bins)
    bins = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # bin centers from aggregated table
    bin_centers = [interval.mid for interval in table_EL.index]

    # ensure numeric
    y_emp = pd.to_numeric(table_EL["Share_EL"], errors="coerce")
    y_con = pd.to_numeric(table_EL_x["Share_EL"], errors="coerce")

    plt.figure(figsize=(7,5))

    # EL share lines
    plt.plot(
        bin_centers,
        y_emp,
        color="#0A84FF",
        marker="o",
        linewidth=2,
        label="Empirical LGD"
    )

    plt.plot(
        bin_centers,
        y_con,
        color="#5AC8FA",
        marker="o",
        linewidth=2,
        label="Constant LGD (15%)"
    )

    # labels
    plt.xlabel("PD Buckets", fontsize=14)
    plt.ylabel("Share of Portfolio EL", fontsize=14)

    plt.title(
        "Expected Loss Concentration Across PD Buckets",
        fontsize=16,
        pad=15
    )

    # ticks
    plt.xticks(bins, rotation=45)

    # percentage formatting on y-axis
    from matplotlib.ticker import PercentFormatter
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    # grids
    plt.grid(axis="x", linestyle="--", alpha=0.4)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    # legend
    plt.legend()

    plt.tight_layout()
    plt.show()

# plot: EL heatmap on PD,LGD

def my_el_heatmap(df):

    ######## create 2D table for heatmap

    # quantile buckets for PD
    df["PD_bin"]  = pd.qcut(df["PD"], 5)

    # fixed buckets for LGD
    lgd_bins = [0, 0.1, 0.3, 0.5, 0.7, 1.0]
    df["LGD_bin"] = pd.cut(df["LossGivenDefault"], bins=lgd_bins)

    # EL total
    total_EL = df["EL"].sum()

    # EL share
    df["EL_share"] = df["EL"] / total_EL

    # look
    #print(df.head())

    # 2D table
    heatmap_table = pd.pivot_table(
        df,
        values="EL_share",
        index="LGD_bin",
        columns="PD_bin",
        aggfunc="sum"
    )

    # choose PD labels 1/3?
    pd_labels_1 = [
        f"{max(interval.left,0)*100:.0f}–{interval.right*100:.0f}%"
        for interval in heatmap_table.columns
    ]
    heatmap_table.columns = pd_labels_1
    # look
    #print(heatmap_table)

    # choose PD labels 2/3?
    pd_labels_2 = [
        "low","medium-low","medium","medium-high","high"
    ]
    heatmap_table.columns = pd_labels_2
    #print(heatmap_table)
    
    # choose PD labels 3/3?
    pd_labels_3 = [
        "≤10%", "10–12%", "12–15%", "15–20%", "≥20%"
    ]
    heatmap_table.columns = pd_labels_3

    # fix LGD labels
    lg_labels = [
        f"{max(interval.left,0)*100:.0f}–{interval.right*100:.0f}%"
        for interval in heatmap_table.index
    ]
    heatmap_table.index = lg_labels

    print(heatmap_table)

    ############ plot heatmap

    plt.figure(figsize=(5,4))

    im = plt.imshow(
        heatmap_table,
        aspect="auto",
        cmap="Blues",
        vmin=0.01,
        vmax=heatmap_table.values.max()
    )

    cbar = plt.colorbar(
        im,
        label="Share of Portfolio EL"
    )

    cbar.ax.yaxis.set_major_formatter(
        mtick.PercentFormatter(1)
    )

    plt.xticks(
        range(len(heatmap_table.columns)),
        heatmap_table.columns.astype(str),
        rotation=45
    )

    plt.yticks(
        range(len(heatmap_table.index)),
        heatmap_table.index.astype(str),
        rotation=45
    )

    # -------- cell annotations ------------
    for i in range(heatmap_table.shape[0]):
        for j in range(heatmap_table.shape[1]):

            value = heatmap_table.iloc[i,j]

            if pd.notna(value):
            
                color = "white" if value > 0.12 else "black"
                plt.text(
                    j,
                    i,
                    f"{value:.1%}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=9
                )

    plt.xlabel("PD Bucket")
    plt.ylabel("LGD Bucket")
    plt.title("Expected Loss Concentration")

    plt.tight_layout()
    plt.show()

# plot: comparison of various calibrations

def my_calibration_comparison(y_true, pd_pred_raw, pd_pred_1, pd_pred_2, pd_pred_3):

    # inputs: 
    #         y_true        observations
    #         pd_pred_raw   predicted PDs, raw/uncalibrated
    #         pd_pred_1     predicted PDs, calibration 1
    #         pd_pred_2     predicted PDs, calibration 2 
    #         pd_pred_3     predicted PDs, calibration 3

    # 1. risk table

    # observations and predictions 
    target   = y_true      # 0/1
    probab   = pd_pred_raw # predicted PDs (raw)
    probab_1 = pd_pred_1   # predicted PDs (Nr1)
    probab_2 = pd_pred_2   # predicted PDs (Nr2)
    probab_3 = pd_pred_3   # predicted PDs (Nr3)

    # define bins 
    # a. equal bins:
    #bins = np.linspace(0, 1, 21)
    # b. manual bins:
    bins = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0]

    df_eval = pd.DataFrame({
        "probability_raw": probab, # predicted PD raw
        "default":         target, # 0/1
        "probability_1":   probab_1, # predicted PD Nr1
        "probability_2":   probab_2, # predicted PD Nr2
        "probability_3":   probab_3, # predicted PD Nr3
    })

    df_eval["prob_bin"] = pd.cut(df_eval["probability_raw"], bins=bins)

    table = (
        df_eval.groupby("prob_bin", observed=False)
        .agg(
            Count=("default", "count"),
            Defaults=("default", "sum"),
            Avg_PD_1=("probability_1", "mean"),
            Avg_PD_2=("probability_2", "mean"),
            Avg_PD_3=("probability_3", "mean")
        )
    )

    # Observed default rate
    table["Observed_Default_Rate"] = table["Defaults"] / table["Count"]

    # Confidence interval (95%)
    table["CI_std"] = np.sqrt(table["Observed_Default_Rate"] * (1 - table["Observed_Default_Rate"]) / table["Count"])

    # high and low
    table["CI_lower"] = table["Observed_Default_Rate"] - 1.96 * table["CI_std"]
    table["CI_upper"] = table["Observed_Default_Rate"] + 1.96 * table["CI_std"]

    # clean up (avoid negatives or >1)
    table["CI_lower"] = table["CI_lower"].clip(lower=0)
    table["CI_upper"] = table["CI_upper"].clip(upper=1)

    # Share of portfolio (BY NUMBER OF LOANS)
    table["Share_of_Portfolio"] = table["Count"] / table["Count"].sum()

    # print table
    table = table.reset_index()
    display(table.style.hide(axis="index")) 

    # save table:
    # table.to_csv("calibration_table.csv", index=False)

    print("-" * 40)

    # 2. plot

    # Bin centers from interval
    bin_centers = [interval.mid for interval in table["prob_bin"]]

    fig, ax = plt.subplots(figsize=(6,5)) 

    # Calibration lines:
    ax.plot(bin_centers, table["Avg_PD_1"],              color="#E9C46A", marker='o', label="1 recent year ", linewidth=2.0, markersize=5)
    ax.plot(bin_centers, table["Avg_PD_2"],              color="#FA8072", marker='s', label="2 recent years", linewidth=2.0, markersize=5)
    ax.plot(bin_centers, table["Avg_PD_3"],              color="#D1495B", marker='D', label="5 recent years", linewidth=2.0, markersize=5)
    ax.plot(bin_centers, table["Observed_Default_Rate"], color='black'  , marker='X', label="Observed DR   ", linewidth=3.5, markersize=8)
    #ax.plot([0,1],[0,1],'--',                            color='gray'               , label="Perfect Calibration")

    # Confidence interval shading:
    #ax.fill_between(bin_centers, table["CI_lower"], table["CI_upper"], color="#2A9D8F", alpha=0.2, label="95% CI (Observed DR)")

    ax.set_xlabel("PD Bin")
    ax.set_ylabel("Predicted / Observed Default Rate")
    ax.set_title("Recalibration: Short vs Long Historical Windows")
    ax.legend()
    # -> set ticks to bin edges:
    #ax.set_xticks(bins)
    # -> or set ticks manually:
    ax.set_xticks(np.arange(0, 0.8, 0.1))
    ax.set_yticks(np.arange(0, 0.8, 0.1))
    # -> add vertical grid aligned with bins
    ax.grid(axis='x', linestyle='-', alpha=0.25)
    # -> add horizontal grid 
    ax.grid(axis='y', linestyle='-', alpha=0.25)
    # optional: rotate labels if crowded
    #ax.set_xticklabels(bins, rotation=45)
    # limit plot
    ax.set_xlim(0, 0.75)
    ax.set_ylim(0, 0.75)
    # use percentages
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    plt.tight_layout()