import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from IPython.display import display

# plot: pd threshold investigation and plot

def my_thresholds(y_true, pd_pred, pd_thresholds):

    probab = pd_pred
    target = y_true
    thresholds = pd_thresholds

    approval_rates = []
    default_rates  = []    

    # CUT-OFF PRINT

    for t in thresholds:

        approved = probab <= t
    
        approval_rate = approved.sum() / len(target)
        approval_rates.append(approval_rate)

        default_rate = target[approved].mean()
        default_rates.append(default_rate)

        print(
            f"t={t:.2f}, "
            f"approved={approved.sum():5.0f}, "
            f"total={len(y_true)}, "
            f"approval rate={approval_rate:.4f}, "
            f"default rate={default_rate:.4f}"
        )

    # CUT-OFF PLOT

    plt.figure(figsize=(5,4))

    approval = np.array(approval_rates) * 100
    default  = np.array(default_rates) * 100
    plt.plot(approval, default, color='black', marker='o')

    # Annotate each point with cut-off
    for i, cutoff in enumerate(thresholds):
        plt.text(approval[i]+5, default[i],
                 f"{cutoff:.2f}",
                 fontsize=11,
                 verticalalignment='bottom',
                 color='lightskyblue')

    # Baseline line
    baseline_default = 14.8
    plt.axhline(y=baseline_default, linestyle='--', color='black')
    plt.text(max(approval)*0.25, baseline_default-0.05,
             "Baseline default (14.8%)",
             verticalalignment='top')

    # chosen point
    plt.scatter(81.27, 12.23, color='limegreen', marker='o', s=200)

    plt.xlabel("approval (%)")
    plt.ylabel("default (%)")
    plt.title("Threshold Plot")
    #plt.legend()
    plt.grid(True)
    plt.show()

# plot: pd threshold investigation and plot (LinkedIn version)

def my_thresholds_li(y_true, pd_pred, pd_thresholds):

    probab = pd_pred
    target = y_true
    thresholds = pd_thresholds

    approval_rates = []
    default_rates  = []    

    # CUT-OFF PRINT

    for t in thresholds:

        approved = probab <= t
    
        approval_rate = approved.sum() / len(target)
        approval_rates.append(approval_rate)

        default_rate = target[approved].mean()
        default_rates.append(default_rate)

        print(
            f"t={t:.2f}, "
            f"approved={approved.sum():5.0f}, "
            f"total={len(y_true)}, "
            f"approval rate={approval_rate:.4f}, "
            f"default rate={default_rate:.4f}"
        )

    # CUT-OFF PLOT

    plt.figure(figsize=(5,4))

    approval = np.array(approval_rates) * 100
    default  = np.array(default_rates) * 100
    plt.plot(approval, default, color='black', marker='o')

    # Baseline line
    baseline_default = 14.8
    plt.axhline(y=baseline_default, linestyle='-', color='black')
    plt.text(max(approval)*0.25, baseline_default-0.05,
             "Baseline default: 14.8%",
             verticalalignment='top')

    # chosen point
    plt.scatter(81.27, 12.23, color='lightskyblue', marker='o', s=266)
    plt.scatter(81.27, 12.23, color='limegreen'   , marker='o', s=166)

    plt.xlabel("approval (%)" , fontsize=16)
    plt.ylabel("default (%)"  , fontsize=16)
    plt.title("Threshold Plot", fontsize=16)
    #plt.legend()
    plt.grid(True)
    plt.show()

# function: impact tables and plots

def my_impact(y_true, pd_pred, pd_threshold=0.20):

    ### ==== DATAFRAME ===

    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": pd_pred,
    })

    ### ===== CONFIG =====

    BINNING_METHOD = "fixed"                                                    # "fixed" or "quantile"
    N_BINS = 6                                                                  # for quantile
    bins = [0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # for fixed
    BASELINE_COLOR = "#D0D5DD"   # lighter gray
    AFTER_COLOR    = "#00A86B"   # green
    ACCENT_COLOR   = "limegreen" # Bondora-like green
    ALPHA_BASE     = 0.60        # alpha
    ALPHA_AFTER    = 0.95        # alpha

    plt.style.use("default")
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False
    })

    ### ===== THRESHOLD =====

    df["accepted_BEFORE"] = (df["y_pred"] < 1.00).astype(int)
    df["accepted_AFTER"]  = (df["y_pred"] < 0.20).astype(int)

    print("\n--- accepted:")
    print("Total observations:", len(df))
    print("Accepted BEFORE:", df["accepted_BEFORE"].sum())
    print("Accepted AFTER:", df["accepted_AFTER"].sum())
    print(f"Approval rate AFTER: {df['accepted_AFTER'].mean():.2%}")

    ### ===== BINNING =====

    if BINNING_METHOD == "fixed":
        df["prob_bin"] = pd.cut(df["y_pred"], bins=bins, include_lowest=True)

    elif BINNING_METHOD == "quantile":
        df["prob_bin"] = pd.qcut(df["y_pred"], q=N_BINS, duplicates="drop")

    # ensure consistent ordering
    df["prob_bin"] = df["prob_bin"].cat.as_ordered()

    ### ===== IMPACT CALCULATION =====

    # BEFORE
    impact_before = (
        df.groupby("prob_bin")
        .agg(count=("prob_bin", "size"),
             defaults=("y_true", "sum"))
        .reset_index()
    )

    # AFTER
    df_after = df[df["accepted_AFTER"] == 1]
    print(f"Default rate AFTER: {df_after['y_true'].mean():.2%}")

    impact_after = (
        df_after.groupby("prob_bin")
        .agg(count=("prob_bin", "size"),
             defaults=("y_true", "sum"))
        .reset_index()
    )

    # ALIGN BINS
    full_bins = df["prob_bin"].cat.categories

    impact_before_full = (
        impact_before.set_index("prob_bin")
        .reindex(full_bins)
        .fillna(0)
        .rename_axis("prob_bin")
        .reset_index()
    )

    impact_after_full = (
        impact_after.set_index("prob_bin")
        .reindex(full_bins)
        .fillna(0)
        .rename_axis("prob_bin")
        .reset_index()
    )

    # SHARES
    impact_before_full["share_before"] = (
        impact_before_full["count"] / impact_before_full["count"].sum()
    )

    impact_after_full["share_after"] = (
        impact_after_full["count"] / impact_after_full["count"].sum()
    )

    # DEFAULT RATES
    impact_before_full["default_rate_before"] = np.where(
        impact_before_full["count"] > 0,
        impact_before_full["defaults"] / impact_before_full["count"],
        0
    )

    impact_after_full["default_rate_after"] = np.where(
        impact_after_full["count"] > 0,
        impact_after_full["defaults"] / impact_after_full["count"],
        0
    )

    # FINAL TABLE
    table = impact_before_full.merge(
        impact_after_full[["prob_bin", "share_after", "default_rate_after"]],
        on="prob_bin",
        how="left"
    )

    table["share_after"] = table["share_after"].fillna(0)

    # ensure sorted
    table = table.sort_values("prob_bin").reset_index(drop=True)

    # remove empty bins 
    table = table[table["share_before"] > 0]

    print('\n--- final table')
    display(table.style.hide(axis="index"))

    ### ===== PLOTTING =====

    x = np.arange(len(table))*1.2

    # clean labels
    labels = [f"{int(b.right*100)}%" for b in table["prob_bin"]]


    # --- PLOT 1: PORTFOLIO SHARE ---
    fig, ax = plt.subplots(figsize=(5, 4))

    ax.bar(x - 0.2, table["share_before"], width=0.4,
           color=BASELINE_COLOR, label="Baseline", alpha=ALPHA_BASE)

    ax.bar(x + 0.2, table["share_after"], width=0.4,
           color=AFTER_COLOR, label="After Threshold", alpha=ALPHA_AFTER)

    ax.set_xticks(x, labels, rotation=0, ha="center")

    ax.set_ylabel("Portfolio Share")
    ax.set_xlabel("PD Bin")

    ax.grid(axis="y", alpha=0.1)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)

    ax.set_ylim(0, max(table["share_before"]) * 1.6)

    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")

    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    plt.tight_layout()
    plt.show()


    # --- PLOT 2: DEFAULT RATE ---
    fig, ax = plt.subplots(figsize=(5, 4))

    ax.plot(x, table["default_rate_before"],
            marker="o", linewidth=2.0,
            color=BASELINE_COLOR, label="Baseline", alpha=ALPHA_BASE)

    ax.plot(x, table["default_rate_after"],
            marker="o", linewidth=2.0,
            color=AFTER_COLOR, label="After Threshold", alpha=ALPHA_AFTER)

    ax.set_xticks(x, labels, rotation=0, ha="center")

    ax.set_ylabel("Default Rate")
    ax.set_xlabel("PD Bin")

    ax.grid(axis="y", alpha=0.1)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)

    ax.set_ylim(0, max(table["default_rate_before"]) * 1.1)

    ax.legend(frameon=False, loc="upper left")
    ax.set_axisbelow(True)

    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    # default base
    plt.axhline(df["y_true"].mean(), linestyle='--', alpha=0.7, color='black')
    plt.text(4.4, df["y_true"].mean(), "Baseline default (14.8%)", alpha=0.7, verticalalignment='top', fontsize=10)

    plt.tight_layout()
    plt.show()

    # default contribution
    table["default_contribution"] = table["share_before"] * table["default_rate_before"]
    display(table.style.hide(axis="index"))

# function: display EL table

def display_el(table):
    print("-" * 50)

    display(
        table.style.format(
            "{:,.0f}",
            subset=pd.IndexSlice[["Count", "Sum_EL"], :]
        ).format(
            "{:.2%}",
            subset=pd.IndexSlice[["Avg_PD", "Share_EL"], :]
        )
    )
    print("-" * 50)