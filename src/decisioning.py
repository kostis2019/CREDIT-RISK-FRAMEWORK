import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import shap
from IPython.display import display
from src.calibration import intercept_recalibration, PDCalibrator
from src.metrics import calculate_metrics
from src.modelling import apply_pipe_PD, estimate_capital
from src.plots import plot_calibration
from src.utils import display_table, apply_slide_style

# plot: pd threshold investigation and plot

def show_approval_vs_default_on_thresholds(y_true, pd_pred, pd_thresholds, title="Threshold Plot", invert=False):

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

    fig, ax = plt.subplots(figsize=(5, 4))

    approval = np.array(approval_rates) * 100
    default  = np.array(default_rates) * 100

    ax.plot(approval, default, color='black', marker='o')

    # Annotate each point with cut-off
    for i, cutoff in enumerate(thresholds):
        ax.text(
            approval[i] - 3,
            default[i] - 0.0,
            f"{cutoff:.2f}",
            fontsize=7,
            fontweight="bold",
            verticalalignment='bottom',
            color='lightskyblue'
        )

    # Baseline line
    baseline_default = 14.8

    ax.axhline(
        y=baseline_default,
        linestyle='--',
        color='black'
    )

    ax.text(
        max(approval) * 0.5,
        baseline_default - 0.05,
        "Baseline default (14.8%)",
        verticalalignment='top'
    )

    # chosen point
    ax.scatter(
        81.27,
        12.23,
        color='limegreen',
        marker='o',
        s=200
    )

    ax.set_xlabel("approval (%)")
    ax.set_ylabel("default (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.2, linewidth=0.8)

    # 100 -> 0
    if invert:
        ax.invert_xaxis()

    # Apply slide style
    from src.utils import apply_slide_style
    apply_slide_style(ax)

    plt.tight_layout()
    plt.show()

    return fig

# plot: pd threshold investigation and plot (LinkedIn version)

def show_approval_vs_default_on_thresholds_l(y_true, pd_pred, pd_thresholds):

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

def show_impact(y_true, pd_pred, pd_threshold=0.20):

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
    df["accepted_AFTER"]  = (df["y_pred"] < pd_threshold).astype(int)

    print("\n--- ACCEPTANCE BEFORE/AFTER PD THRESHOLD:")
    print("Total observations:", len(df))
    print("Accepted BEFORE:", df["accepted_BEFORE"].sum())
    print("Accepted AFTER: ", df["accepted_AFTER"].sum())
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

    #print('\n--- final table')
    #display_table(table)

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
    display_table(table)

# function: aggregate EL on PD bins

def aggregate_el_on_pd(df_with_el, el_column, bins):
    
    total_EL = df_with_el[el_column].sum()

    # evaluation data and bin
    df_eval = df_with_el[["PD", el_column]].copy()
    df_eval["prob_bin"] = pd.cut(df_eval["PD"], bins=bins)

    # aggregation
    table_EL = (
        df_eval.groupby("prob_bin", observed=False)
        .agg(
            Count  = ("PD"     , "count"),
            Avg_PD = ("PD"     , "mean"),
            Sum_EL = (el_column, "sum")
        )
    )

    # EL share
    table_EL["Share_EL"] = table_EL["Sum_EL"] / total_EL

    return table_EL

# function: aggregate DR on years

def aggregate_dr_on_years(ds, date_column, default_column, year_column="Year"):

    # create year
    ds[year_column] = ds[date_column].dt.year

    # group by year and aggregate
    dr_summary = ds.groupby(year_column).agg(
        num_loans    = (year_column   , "count"),
        num_defaults = (default_column, "sum")
    ).reset_index()

    # calculate default rate
    dr_summary["default_rate"] = dr_summary["num_defaults"] / dr_summary["num_loans"]

    return dr_summary

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

# process: compare calibration windows

def compare_calibration_windows(X_sample, y_sample, calibration_windows, pipeline, show_cali_plots=True):

    results = {}

    for name, run_data in calibration_windows.items():

        print("=" * 70)
        print(f"Run name: {name}")
        print("=" * 70)

        ###################################################
        # calibration training sample
        ###################################################

        print("-" * 40)
        print(f"applying pipeline...")

        predictions_cal   = apply_pipe_PD(run_data["X"], pipeline)

        print("-" * 40)
        print(f"training calibration...")

        y_pred_cal, delta = intercept_recalibration(run_data["y"], predictions_cal["PD"])
        print(f"Intercept shift (delta): {delta:.6f}")

        ###################################################
        # sample to apply (before calibration)
        ###################################################

        print("-" * 40)
        print("BEFORE CALIBRATION")
        print("-" * 40)
    
        print("-" * 40)
        print(f"applying pipeline...")

        predictions_sample = apply_pipe_PD(X_sample, pipeline, column_lgd="LossGivenDefault", column_ead="Amount", calculate_el=True)

        if show_cali_plots:
            plot_calibration(y_sample, predictions_sample["PD"], show_cal_table=False, plot_title="uncalibrated sample", bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        metrics_before = calculate_metrics(
            y_true=y_sample,
            pd_pred=predictions_sample["PD"],
            exposure=predictions_sample["Amount"],
            el=predictions_sample["EL"],
            dataset_name=f"OOT ({name})",
            verbose=False,
        )

        ###################################################
        # train calibration
        ###################################################

        calibrator = PDCalibrator(method="shift", shift=delta)
        calibrator.fit(predictions_cal, run_data["y"])

        ###################################################
        # calibrate sample
        ###################################################

        print("-" * 40)
        print(f"calibrating sample...")

        predictions_sample = calibrator.transform(predictions_sample)
        predictions_sample["EL"] = (predictions_sample["PD"] * predictions_sample["Amount"] * predictions_sample["LossGivenDefault"])

        ###################################################
        # sample to apply (after calibration)
        ###################################################

        print("-" * 40)
        print("AFTER CALIBRATION")
        print("-" * 40)

        if show_cali_plots:
            plot_calibration(y_sample, predictions_sample["PD"], show_cal_table=False, plot_title="calibrated sample", bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        metrics_after = calculate_metrics(
            y_true=y_sample,
            pd_pred=predictions_sample["PD"],
            exposure=predictions_sample["Amount"],
            el=predictions_sample["EL"],
            dataset_name=f"OOT ({name})",
            verbose=False,
        )

        ###################################################
        # save
        ###################################################

        results[name] = {
            "delta": delta,
            "predictions": predictions_sample.copy(),
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
        }

    return results

# process: compare calibration methods

def compare_calibration_methods(X_sample, y_sample, calibration_methods, pipeline, show_cali_plots=True):

    results = {}

    for name, run_data in calibration_methods.items():

        print("=" * 70)
        print(f"Run name: {name}")
        print("=" * 70)

        ###################################################
        # calibration training sample
        ###################################################

        print("-" * 40)
        print(f"applying pipeline...")

        predictions_cal   = apply_pipe_PD(run_data["X"], pipeline)

        print("-" * 40)
        print(f"training calibration...")

        # step needed only when "shift"
        if run_data["method"] == "shift":
            y_pred_cal, delta = intercept_recalibration(run_data["y"], predictions_cal["PD"])
            print(f"Intercept shift (delta): {delta:.6f}")

        ###################################################
        # sample to apply (before calibration)
        ###################################################

        print("-" * 40)
        print("BEFORE CALIBRATION")
        print("-" * 40)
    
        print("-" * 40)
        print(f"applying pipeline...")

        predictions_sample = apply_pipe_PD(X_sample, pipeline, column_lgd="LossGivenDefault", column_ead="Amount", calculate_el=True)

        if show_cali_plots:
            plot_calibration(y_sample, predictions_sample["PD"], show_cal_table=False, plot_title="uncalibrated sample", bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        metrics_before = calculate_metrics(
            y_true=y_sample,
            pd_pred=predictions_sample["PD"],
            exposure=predictions_sample["Amount"],
            el=predictions_sample["EL"],
            dataset_name=f"OOT ({name})",
            verbose=False,
        )

        ###################################################
        # train calibration
        ###################################################

        if run_data["method"] == "shift":
            calibrator = PDCalibrator(method="shift", shift=delta)
        else:
            calibrator = PDCalibrator(method=run_data["method"])
            
        # fit
        calibrator.fit(predictions_cal, run_data["y"])

        ###################################################
        # calibrate sample
        ###################################################

        print("-" * 40)
        print(f"calibrating sample...")

        predictions_sample = calibrator.transform(predictions_sample)
        predictions_sample["EL"] = (predictions_sample["PD"] * predictions_sample["Amount"] * predictions_sample["LossGivenDefault"])

        ###################################################
        # sample to apply (after calibration)
        ###################################################

        print("-" * 40)
        print("AFTER CALIBRATION")
        print("-" * 40)

        if show_cali_plots:
            plot_calibration(y_sample, predictions_sample["PD"], show_cal_table=False, plot_title="calibrated sample", bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        metrics_after = calculate_metrics(
            y_true=y_sample,
            pd_pred=predictions_sample["PD"],
            exposure=predictions_sample["Amount"],
            el=predictions_sample["EL"],
            dataset_name=f"OOT ({name})",
            verbose=False,
        )

        ###################################################
        # save
        ###################################################

        results[name] = {
            "delta": delta,
            "predictions": predictions_sample.copy(),
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
        }

    return results

# process: estimate EL and CAP over different PD thresholds

def estimate_capital_reduction(df, thresholds, column_lgd, column_def, column_ead, verbose=False):

    from src.modelling import estimate_capital

    probab = df["PD"]

    pd_thresholds      = []
    approval_rates     = []
    portfolio_ead      = []
    expected_loss_det  = []
    expected_loss_mc   = []
    economic_capital   = []
    economic_capital_reduction = []

    # LOOP OVER PD THRESHOLDS

    for t in thresholds:

        approved = probab <= t
        approval_rate = approved.sum() / len(df)

        df_approved = df[approved]

        print('running Monte-Carlo... ', 'PD threshold: ', t)
        result_df, loss_sum, el_sum, _ = estimate_capital(df_approved,  
                                                          method="monte-carlo", 
                                                          column_lgd=column_lgd, 
                                                          column_def=column_def, 
                                                          column_ead=column_ead, 
                                                          allocate=False, 
                                                          verbose=False)

        pd_thresholds.append(t)   
        approval_rates.append(approval_rate)     
        portfolio_ead.append(df_approved["Amount"].sum())  
        expected_loss_det.append(result_df["EL"].sum())  
        expected_loss_mc.append(el_sum["EL (MC-simulated)"])   
        economic_capital.append(loss_sum["Economic Capital"])

    results = pd.DataFrame({
        "PD Threshold":       pd_thresholds,
        "Approval Rate":      approval_rates,
        "Portfolio EAD":      portfolio_ead,
        "Expected Loss":      expected_loss_det,
        "Expected Loss (MC)": expected_loss_mc,
        "Economic Capital":   economic_capital,
        })

    results["EL Reduction"] = (1 - results["Expected Loss (MC)"] / results.loc[0, "Expected Loss (MC)"])
    results["EC Reduction"] = (1 - results["Economic Capital"]   / results.loc[0, "Economic Capital"])
    
    results_display = results.copy()
    results_display["Portfolio EAD"]      /= 1e6
    results_display["Expected Loss"]      /= 1e6
    results_display["Expected Loss (MC)"] /= 1e6
    results_display["Economic Capital"]   /= 1e3
    results_display["EL Reduction"]       *= 1e2
    results_display["EC Reduction"]       *= 1e2

    if verbose:
        display(
            results_display.style
            .format({
                "PD Threshold":       "{:.2f}",
                "Approval Rate":      "{:.1%}",
                "Portfolio EAD":      "{:.1f} M€",
                "Expected Loss":      "{:.2f} M€",
                "Expected Loss (MC)": "{:.2f} M€",
                "Economic Capital":   "{:.0f} k€",
                "EL Reduction":       "{:.0f} %",
                "EC Reduction":       "{:.0f} %",        
            })
            .hide(axis="index")
            )      

    return results

# plot: pd threshold investigation and plot (LinkedIn version)

def show_approval_vs_reduction(table, show_plot=True, invert_y=True):

    expected   = table["EL Reduction"]
    capital    = table["EC Reduction"]
    approval   = table["Approval Rate"]
    thresholds = table["PD Threshold"]   

    # PLOT

    fig, ax = plt.subplots(figsize=(5, 4))

    approval   = np.array(approval) * 100
    expected   = np.array(expected) * 100
    capital    = np.array(capital)  * 100
    thresholds = np.array(thresholds)

    ax.plot(approval, expected, color="#B39DDB", marker='o', label='Expected Loss')
    ax.plot(approval, capital , color="#512DA8", marker='o', label='Economic Capital')

    ax.invert_xaxis()
    if invert_y:
        ax.invert_yaxis()

    ax.set_xlabel("approval (%)" , fontsize=10)
    ax.set_ylabel("reduction (%)", fontsize=10)
    ax.set_title("Risk Reduction", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.2, linewidth=0.8)

    # top x-axis: pd threshold

    threshold_color = "lightskyblue"
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(approval)
    ax_top.set_xticklabels([f"{x:.2f}" for x in thresholds], fontweight="bold", color = threshold_color)
    ax_top.tick_params(axis='x', labelsize=7, pad=2, color = threshold_color)
    ax_top.set_xlabel("PD Threshold", fontsize=10, labelpad=4, fontweight="bold", color = threshold_color) 

    plt.tight_layout()

    # Apply slide style
    from src.utils import apply_slide_style
    apply_slide_style(ax)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig

# process: explain model 

def explain_model(pipeline, X_raw):

    # extract base model
    calibrated_model = pipeline.named_steps["model"]
    base_model = calibrated_model.model_
    model_type = type(base_model).__name__
    feature_names = pipeline.named_steps["feature_selection"].selected_features

    print("Model type: ", model_type)

    if model_type == "LogisticRegression":

        coefficients = base_model.coef_[0]

        explanation = pd.DataFrame({
            "feature": feature_names,
            "coefficient": coefficients,
            "abs_coefficient": abs(coefficients)
        }).sort_values(
            "abs_coefficient", ascending=False
        )

        # plot:

        fig, ax = plt.subplots(figsize=(7.45, 5))

        colors = ["blue" if x > 0 else "red" for x in explanation["coefficient"]]

        ax.barh(
            explanation["feature"],
            explanation["coefficient"],
            height=0.6,
            color=colors,
            alpha=0.60)

        ax.axvline(0, color="black", linewidth=0.8)
        ax.invert_yaxis()
        ax.set_xlabel("Coefficient")
        ax.set_ylabel("")
        plt.tight_layout()

        # show plot now!
        display(fig)
        plt.close(fig)

    elif model_type in ["GradientBoostingClassifier", "XGBClassifier"]:

        if X_raw is None:
            raise ValueError(
                "X must be provided for SHAP explainability."
            )

        X_explain = X_raw

        # using the pre-processing of the pipeline to create the X_explain
        # X_explain is the dataset to run shap on !!
        for name, step in pipeline.named_steps.items():
            if name == "model":
                break
                
            X_explain = step.transform(X_explain)

        # run shap here !!
        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(X_explain)

        # average over observations
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

        explanation = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap
        }).sort_values(
            "mean_abs_shap", ascending=False
        )

        # plot:

        # 1: mean abs shap
        # 2: beeswarm
        which_plot = 2

        fig, ax = plt.subplots(figsize=(7.45, 5))

        if which_plot == 1:

            ax.barh(
                explanation["feature"],
                explanation["mean_abs_shap"],
                height=0.6,
                color="#FF7A00",
                alpha=0.60)

            ax.axvline(0, color="black", linewidth=0.8)
            ax.invert_yaxis()
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_ylabel("")
            plt.tight_layout()

        if which_plot == 2:

            shap.summary_plot(
            shap_values,
            X_explain,
            feature_names=feature_names,
            plot_type="dot",
            show=False)

            fig = plt.gcf()
            fig.set_size_inches(6, 5)

        # show plot now!
        display(fig)
        plt.close(fig)

    else:
        raise ValueError(
            f"Explainability not implemented for {model_type}"
        )

# process: estimate EL and CAP over different PD stress factors

def estimate_reverse_capital_stress(df, threshold, stress_factors, column_lgd, column_def, column_ead, verbose=False, seed=42):

    # initial PDs (un-stressed)
    probab = df["PD"]

    stress_factor      = []
    mean_pd            = []
    portfolio_ead      = []
    expected_loss_det  = []
    expected_loss_mc   = []
    economic_capital   = []

    loss_mean            = []
    loss_dev             = []
    loss_var95           = []
    loss_var999          = []
    loss_dev_theoretical = []

    seeds                = []

    # DEFINE APPROVED PORTFOLIO

    approved = probab <= threshold
    df_approved = df.loc[approved].copy()

    # LOOP OVER PD STRESS FACTORS

    for s in stress_factors:

        # always start from the original PDs
        df_stressed = df_approved.copy()

        # stress pds
        df_stressed["PD"] = (df_approved["PD"] * s).clip(upper=1.0)

        # run monte-carlo: print
        if verbose:
            print('running Monte-Carlo... ', 'PD stress factor: ', s)
            print(' PD ', ' min: ', f'{df_stressed["PD"].min():.4f}', ' mean: ', f'{df_stressed["PD"].mean():.4f}', ' max: ', f'{df_stressed["PD"].max():.4f}')
        
        # run monte-carlo
        result_df, loss_sum, el_sum, _ = estimate_capital(df_stressed,  
                                                          method="monte-carlo", 
                                                          column_lgd=column_lgd, 
                                                          column_def=column_def, 
                                                          column_ead=column_ead, 
                                                          allocate=False, 
                                                          verbose=False,
                                                          seed=seed)

        # calculate theoretical loss dev
        loss_dev_theo = np.sqrt(
            (
                (df[column_ead] * df[column_lgd]) ** 2
                * df_stressed["PD"]
                * (1 - df_stressed["PD"])
            ).sum()
        )

        # append results
        stress_factor.append(s)
        mean_pd.append(df_stressed["PD"].mean())        
        portfolio_ead.append(df_stressed[column_ead].sum())  
        expected_loss_det.append(result_df["EL"].sum())  
        expected_loss_mc.append(el_sum["EL (MC-simulated)"])   
        economic_capital.append(loss_sum["Economic Capital"])
        loss_mean.append(loss_sum["Loss (mean)"])
        loss_dev.append(loss_sum["Loss (std)"])
        loss_var95.append(loss_sum["VaR 95%"])
        loss_var999.append(loss_sum["VaR 99.9%"])
        loss_dev_theoretical.append(loss_dev_theo)
        seeds.append(seed)

    results = pd.DataFrame({
        "PD Threshold":        threshold,
        "Stress Factor":       stress_factor,
        "Mean PD":             mean_pd,
        "Portfolio EAD":       portfolio_ead,
        "Expected Loss":       expected_loss_det,
        "Expected Loss (MC)":  expected_loss_mc,
        "Economic Capital":    economic_capital,
        "Loss (mean)":         loss_mean,
        "Loss (dev)":          loss_dev,
        "VaR 95%":             loss_var95,
        "VaR 99.9%":           loss_var999,
        "Loss (dev) (theory)": loss_dev_theoretical,
        "Seed":                seed
        })
    
    results_display = results.copy()
    results_display["Portfolio EAD"]      /= 1e6
    results_display["Expected Loss"]      /= 1e6
    results_display["Expected Loss (MC)"] /= 1e6
    results_display["Economic Capital"]   /= 1e3

    if verbose:
        display(
            results_display.style
            .format({
                "PD Threshold":        "{:.2f}",
                "Stress Factor":       "{:.2f}",
                "Mean PD":             "{:.2f}",
                "Portfolio EAD":       "{:.1f} M€",
                "Expected Loss":       "{:.2f} M€",
                "Expected Loss (MC)":  "{:.2f} M€",
                "Economic Capital":    "{:.0f} k€",
                "Loss (mean)":         "{:.0f}",
                "Loss (dev)":          "{:.0f}",
                "VaR 95%":             "{:.0f}",
                "VaR 99.9%":           "{:.0f}",
                "Loss (dev) (theory)": "{:.0f}",
            })
            .hide(axis="index")
            )      

    return results

# show plots of reverse monte-carlo process

def show_pd_stress_vs_capital(table, show_plot=True, invert_y=True):

    capital       = table["Economic Capital"]
    expected_loss = table["Expected Loss"]
    pd_stress     = table["Stress Factor"]
    mean_pd       = table["Mean PD"]

    capital       = np.array(capital) / 1000           # (k€)
    expected_loss = np.array(expected_loss) / 1000000  # (M€)
    pd_stress     = np.array(pd_stress)
    mean_pd       = np.array(mean_pd)

    # Check if multiple seed experiment
    has_seeds = "Seed" in table.columns

    # PLOT

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

    # Economic Capital

    if has_seeds:

        capital_curves = []

        for seed, group in table.groupby("Seed"):

            group = group.sort_values("Stress Factor")

            x = group["Stress Factor"].to_numpy()
            y = group["Economic Capital"].to_numpy() / 1000

            capital_curves.append(y)

            ax1.plot(
                x, y,
                color="grey",
                alpha=0.30,
                linewidth=0.7
            )

        capital_mean = np.mean(capital_curves, axis=0)

        ax1.plot(
            x, capital_mean,
            color="#512DA8",
            linewidth=2,
            label="Mean Economic Capital"
        )

    else:

        ax1.plot(
            pd_stress, capital,
            color="#512DA8",
            label="Economic Capital"
        )

    ax1.xaxis.set_major_locator(MultipleLocator(0.5))
    ax1.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    if invert_y:
        ax1.invert_yaxis()

    ax1.set_xlabel("PD stress factor", fontsize=10)
    ax1.set_ylabel("Economic Capital (k€)", fontsize=10, color="#512DA8")
    ax1.grid(True, alpha=0.2, linewidth=0.8)

    # Expected Loss

    if has_seeds:

        loss_curves = []

        for seed, group in table.groupby("Seed"):

            group = group.sort_values("Stress Factor")

            x = group["Stress Factor"].to_numpy()
            y = group["Expected Loss"].to_numpy() / 1000000

            loss_curves.append(y)

            ax2.plot(
                x, y,
                color="grey",
                alpha=0.20,
                linewidth=0.7
            )

        loss_mean = np.mean(loss_curves, axis=0)

        ax2.plot(
            x, loss_mean,
            color="#B39DDB",
            linewidth=2,
            label="Expected Loss"
        )

    else:

        ax2.plot(
            pd_stress, expected_loss,
            color="#B39DDB",
            label="Expected Loss"
        )

    ax2.xaxis.set_major_locator(MultipleLocator(0.5))
    ax2.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    if invert_y:
        ax2.invert_yaxis()

    ax2.set_xlabel("PD stress factor", fontsize=10)
    ax2.set_ylabel("Expected Loss (M€)", fontsize=10, color="#B39DDB")
    ax2.grid(True, alpha=0.2, linewidth=0.8)

    # Top x-axis: mean PD

    this_color = "black"
    top_ticks = np.arange(1.0, 5.01, 0.5)

    mean_pd_ticks = np.interp(top_ticks, pd_stress, mean_pd)

    for ax in [ax1, ax2]:

        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xticks(top_ticks)

        ax_top.set_xticklabels(
            [f"{x:.2f}" for x in mean_pd_ticks],
            fontweight="bold",
            color=this_color
        )

        ax_top.tick_params(
            axis="x", labelsize=7, pad=2, color=this_color
        )

        ax_top.set_xlabel(
            "Mean PD",
            fontsize=10,
            labelpad=4,
            fontweight="bold",
            color=this_color
        )

    # Legend

    ax1.legend(loc="lower right")
    ax2.legend(loc="lower right")

    plt.tight_layout()

    # Apply slide style

    from src.utils import apply_slide_style

    apply_slide_style(ax1)
    apply_slide_style(ax2)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig

# show plots of reverse monte-carlo process 

def show_pd_stress_vs_lossdev(table, show_plot=True, invert_y=True):

    loss_mean     = table["Loss (mean)"]
    loss_var95    = table["VaR 95%"]
    loss_var999   = table["VaR 99.9%"]
    loss_dev      = table["Loss (dev)"]
    loss_dev_theo = table["Loss (dev) (theory)"]
    pd_stress     = table["Stress Factor"]
    mean_pd       = table["Mean PD"]

    loss_mean     = np.array(loss_mean) / 1000000
    loss_var95    = np.array(loss_var95) / 1000000
    loss_var999   = np.array(loss_var999) / 1000000
    loss_dev      = np.array(loss_dev) / 1000
    loss_dev_theo = np.array(loss_dev_theo) / 1000
    pd_stress     = np.array(pd_stress)
    mean_pd       = np.array(mean_pd)

    # Check if multiple seed experiment

    has_seeds = "Seed" in table.columns

    # PLOT

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3))

    # --------------------------------------------------
    # Plot 1: Loss standard deviation
    # --------------------------------------------------

    if has_seeds:

        dev_curves = []

        for seed, group in table.groupby("Seed"):

            group = group.sort_values("Stress Factor")

            x = group["Stress Factor"].to_numpy()
            y = group["Loss (dev)"].to_numpy() / 1000

            dev_curves.append(y)

            ax1.plot(
                x, y,
                color="grey",
                alpha=0.30,
                linewidth=0.7
            )

        dev_mean = np.mean(dev_curves, axis=0)

        ax1.plot(
            x, dev_mean,
            color="black",
            linewidth=1.8,
            label="Loss StdDev (mean simulation)"
        )

        # Theoretical SD: same across seeds

        theo_mean = (
            table.groupby("Stress Factor")["Loss (dev) (theory)"]
            .mean()
        )

        ax1.plot(
            theo_mean.index,
            theo_mean.to_numpy() / 1000,
            color="black",
            linestyle="--",
            label="Loss StdDev (theory)"
        )

    else:

        ax1.plot(
            pd_stress, loss_dev_theo,
            color="black",
            label="Loss StdDev (theory)",
            linestyle="--"
        )

        ax1.plot(
            pd_stress, loss_dev,
            color="black",
            label="Loss StdDev (simulation)",
            linestyle="-"
        )

    if invert_y:
        ax1.invert_yaxis()

    ax1.set_xlabel("PD stress factor", fontsize=10)
    ax1.set_ylabel("Loss StDev (k€)", fontsize=10)
    ax1.grid(True, alpha=0.2, linewidth=0.8)

    # --------------------------------------------------
    # Plot 2: Mean loss and VaR
    # --------------------------------------------------

    if has_seeds:

        var95_curves = []
        var999_curves = []

        for seed, group in table.groupby("Seed"):

            group = group.sort_values("Stress Factor")

            x = group["Stress Factor"].to_numpy()

            y95 = group["VaR 95%"].to_numpy() / 1000000
            y999 = group["VaR 99.9%"].to_numpy() / 1000000

            var95_curves.append(y95)
            var999_curves.append(y999)

            ax2.plot(
                x, y95,
                color="grey",
                alpha=0.20,
                linewidth=0.7
            )

            ax2.plot(
                x, y999,
                color="grey",
                alpha=0.30,
                linewidth=0.7
            )

        var95_mean = np.mean(var95_curves, axis=0)
        var999_mean = np.mean(var999_curves, axis=0)

        # Deterministic expected loss

        loss_mean_group = (
            table.groupby("Stress Factor")["Expected Loss"]
            .mean()
        )

        ax2.plot(
            loss_mean_group.index,
            loss_mean_group.to_numpy() / 1000000,
            color="black",
            label="Loss Mean",
            linestyle="-"
        )

        ax2.plot(
            x, var95_mean,
            color="black",
            label="Loss VaR 95 (mean)",
            linestyle="--",
            linewidth=1.5
        )

        ax2.plot(
            x, var999_mean,
            color="black",
            label="Loss VaR 99.9 (mean)",
            linestyle="-.",
            linewidth=1.5
        )

    else:

        ax2.plot(
            pd_stress, loss_mean,
            color="black",
            label="Loss Mean",
            linestyle="-"
        )

        ax2.plot(
            pd_stress, loss_var95,
            color="black",
            label="Loss VaR 95",
            linestyle="--"
        )

        ax2.plot(
            pd_stress, loss_var999,
            color="black",
            label="Loss VaR 99.9",
            linestyle="-."
        )

    if invert_y:
        ax2.invert_yaxis()

    ax2.set_xlabel("PD stress factor", fontsize=10)
    ax2.set_ylabel("Loss (M€)", fontsize=10)
    ax2.grid(True, alpha=0.2, linewidth=0.8)

    # --------------------------------------------------
    # Plot 3: EC / EL and EC / Loss SD
    # --------------------------------------------------

    if has_seeds:

        ratio1_curves = []
        ratio2_curves = []

        for seed, group in table.groupby("Seed"):

            group = group.sort_values("Stress Factor")

            x = group["Stress Factor"].to_numpy()

            ec = group["Economic Capital"].to_numpy()
            el = group["Expected Loss"].to_numpy()
            sd = group["Loss (dev)"].to_numpy()

            r1 = ec / el
            r2 = ec / sd

            ratio1_curves.append(r1)
            ratio2_curves.append(r2)

            ax3.plot(
                x, r1,
                color="grey",
                alpha=0.25,
                linewidth=0.7
            )

            ax3.plot(
                x, r2,
                color="grey",
                alpha=0.25,
                linewidth=0.7
            )

        ratio1_mean = np.mean(ratio1_curves, axis=0)
        ratio2_mean = np.mean(ratio2_curves, axis=0)

        ax3.plot(
            x, ratio1_mean,
            color="black",
            label="EC / EL (mean)",
            linestyle="-."
        )

        ax3.plot(
            x, ratio2_mean,
            color="black",
            label="EC / Loss StdDev (mean)",
            linestyle=":"
        )

    else:

        ratio_1 = (loss_var999 - loss_mean) / loss_mean
        ratio_2 = (loss_var999 - loss_mean) * 1000 / loss_dev

        ax3.plot(
            pd_stress, ratio_1,
            color="black",
            label="EC / EL",
            linestyle="-."
        )

        ax3.plot(
            pd_stress, ratio_2,
            color="black",
            label="EC / Loss StdDev",
            linestyle=":"
        )

    if invert_y:
        ax3.invert_yaxis()

    ax3.set_xlabel("PD stress factor", fontsize=10)
    ax3.set_ylabel("Dimensionless ratio", fontsize=10)
    ax3.grid(True, alpha=0.2, linewidth=0.8)

    # --------------------------------------------------
    # X axes: PD stress factor
    # --------------------------------------------------

    for ax in [ax1, ax2, ax3]:

        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    # --------------------------------------------------
    # Top x-axis: Mean PD
    # --------------------------------------------------

    top_ticks = np.arange(1.0, 5.01, 0.5)

    mean_pd_group = (
        table.groupby("Stress Factor")["Mean PD"]
        .mean()
    )

    mean_pd_ticks = np.interp(
        top_ticks,
        mean_pd_group.index.to_numpy(),
        mean_pd_group.to_numpy()
    )

    for ax in [ax1, ax2, ax3]:

        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xticks(top_ticks)

        ax_top.set_xticklabels(
            [f"{x:.2f}" for x in mean_pd_ticks],
            fontweight="bold",
            color="black"
        )

        ax_top.tick_params(
            axis="x",
            labelsize=7,
            pad=2,
            color="black"
        )

        ax_top.set_xlabel(
            "Mean PD",
            fontsize=10,
            labelpad=4,
            fontweight="bold",
            color="black"
        )

    # --------------------------------------------------
    # Legends and styling
    # --------------------------------------------------

    for ax in [ax1, ax2, ax3]:
        ax.legend(loc="lower right")

    plt.tight_layout()

    apply_slide_style(ax1)
    apply_slide_style(ax2)
    apply_slide_style(ax3)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig

# EC summary for seeds experiments

def ec_summary(results_all, print_table=True):

    # --------------------------------------------------
    # Aggregate Monte Carlo results across seeds
    # --------------------------------------------------

    metrics = [
        "Economic Capital",
        "VaR 99.9%",
        "VaR 95%",
        "Loss (dev)",
        "Loss (mean)"
    ]

    summary = (
        results_all
        .groupby("Stress Factor")[metrics]
        .agg(["mean", "std", "min", "max"])
    )

    # Flatten column names
    summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in summary.columns
    ]

    # --------------------------------------------------
    # Coefficient of variation
    # --------------------------------------------------

    for metric in metrics:

        summary[f"{metric}_CV"] = (
            summary[f"{metric}_std"] /
            summary[f"{metric}_mean"]
        )

    # --------------------------------------------------
    # 5th and 95th percentiles across seeds
    # --------------------------------------------------

    for metric in ["Economic Capital", "VaR 99.9%"]:

        quantiles = (
            results_all
            .groupby("Stress Factor")[metric]
            .quantile([0.05, 0.95])
            .unstack()
        )

        summary[f"{metric}_p05"] = quantiles[0.05]
        summary[f"{metric}_p95"] = quantiles[0.95]

    # --------------------------------------------------
    # Executive summary
    # --------------------------------------------------

    ec_mean = summary["Economic Capital_mean"]
    var_mean = summary["VaR 99.9%_mean"]

    peak_stress = ec_mean.idxmax()
    peak_ec = ec_mean.max()

    baseline_ec = ec_mean.iloc[0]
    final_ec = ec_mean.iloc[-1]

    baseline_el = results_all.groupby(
        "Stress Factor"
    )["Expected Loss"].mean().iloc[0]

    final_el = results_all.groupby(
        "Stress Factor"
    )["Expected Loss"].mean().iloc[-1]

    baseline_var = var_mean.iloc[0]
    final_var = var_mean.iloc[-1]

    n_seeds = results_all["Seed"].nunique()

    print("\n========== MONTE CARLO STRESS SUMMARY ==========\n")

    print(f"Number of seeds:             {n_seeds}")
    print(f"Stress scenarios:            {len(summary)}")

    print("\n--- Expected Loss ---")

    print(f"Baseline EL:                 €{baseline_el:,.0f}")
    print(f"Final EL:                    €{final_el:,.0f}")
    print(f"EL increase:                 {(final_el / baseline_el - 1)*100:.1f}%")

    print("\n--- Economic Capital ---")

    print(f"Baseline EC:                 €{baseline_ec:,.0f}")
    print(f"Maximum mean EC:             €{peak_ec:,.0f}")
    print(f"Peak stress factor:          {peak_stress:.1f}x")
    print(f"Final EC:                    €{final_ec:,.0f}")
    print(f"EC change from peak:         {(final_ec / peak_ec - 1)*100:.1f}%")

    print("\n--- VaR 99.9% ---")

    print(f"Baseline VaR:                €{baseline_var:,.0f}")
    print(f"Final VaR:                   €{final_var:,.0f}")
    print(f"Baseline VaR CV:             {summary['VaR 99.9%_CV'].iloc[0]*100:.2f}%")
    print(f"Final VaR CV:                {summary['VaR 99.9%_CV'].iloc[-1]*100:.2f}%")

    print("\n--- EC variability ---")

    print(f"Baseline EC CV:              {summary['Economic Capital_CV'].iloc[0]*100:.2f}%")
    print(f"Peak EC CV:                  {summary.loc[peak_stress, 'Economic Capital_CV']*100:.2f}%")

    print("\n================================================\n")

    if print_table:
        display(summary.round(3))

    return summary
