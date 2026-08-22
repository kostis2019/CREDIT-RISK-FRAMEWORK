import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    brier_score_loss,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
)

# function: KS statistic

def calculate_ks_statistic(y_true, y_pred_proba):
    # Calculate False Positive Rate (FPR) and True Positive Rate (TPR)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    
    # The KS statistic is the maximum difference between TPR and FPR
    ks_statistic = max(tpr - fpr)
    return ks_statistic

# function: my metrics

def my_metrics(y_true, pd_pred, exposure=None, el=None, dataset_name="", verbose=True):
    
    # -------------------------
    # discrimination
    # -------------------------
    auc = roc_auc_score(y_true, pd_pred)
    fpr, tpr, _ = roc_curve(y_true, pd_pred)
    ks = np.max(tpr - fpr)

    # -------------------------
    # calibration
    # -------------------------
    obs_dr  = np.mean(y_true)
    mean_pd = np.mean(pd_pred)
    brier = brier_score_loss(y_true, pd_pred)

    # -------------------------
    # portfolio
    # -------------------------
    exp_total = np.nan
    el_total  = np.nan
    el_rate   = np.nan
    if exposure is not None:
        exp_total = np.sum(exposure)
    if el is not None:
        el_total = np.sum(el)
    if exposure is not None and el is not None:
        el_rate = el_total / exp_total

    # -------------------------
    # dictionary
    # -------------------------
    metrics = {
        "Dataset"     : dataset_name,
        "AUC"         : auc,
        "KS"          : ks,
        "Observed_DR" : obs_dr,
        "Mean_PD"     : mean_pd,
        "Brier"       : brier,
        "Exposure"    : exp_total,
        "EL_Total"    : el_total,
        "EL_Rate"     : el_rate
    }

    # -------------------------
    # optional print
    # -------------------------
    if verbose:
        print("-" * 40)
        print(dataset_name)
        print("-" * 40)
        print(f"AUC         : {auc:.4f}")
        print(f"KS          : {ks:.4f}")
        print(f"Observed DR : {obs_dr:.2%}")
        print(f"Mean PD     : {mean_pd:.2%}")
        print(f"Brier       : {brier:.4f}")
        if exposure is not None:
            print(f"Exposure    : {exp_total:,.0f}")
        if el is not None:
            print(f"EL Total    : {el_total:,.0f}")
        if exposure is not None and el is not None:
            print(f"EL Rate     : {el_rate:.2%}")
        print("-" * 40)

    return metrics

# function: my threshold metrics

def my_threshold_metrics(y_true, pd_pred, threshold, exposure, el, dataset_name="", verbose=True):

        # Accepted Portfolio       
        accept_all   = pd_pred < 1.00
        accept_thr   = pd_pred < threshold

        # Mean PD       
        accept_all_mean_pd  = pd_pred[accept_all].mean()
        accept_thr_mean_pd  = pd_pred[accept_thr].mean()

        # Acceptance Rate
        accept_all_rate     = accept_all.mean()
        accept_thr_rate     = accept_thr.mean()

        # Exposure    
        accept_all_amt      = exposure[accept_all].sum()
        accept_thr_amt      = exposure[accept_thr].sum()

        # Mean Exposure
        accept_all_mean_amt = exposure[accept_all].mean()
        accept_thr_mean_amt = exposure[accept_thr].mean()

        # EL
        accept_all_el       = el[accept_all].sum()
        accept_thr_el       = el[accept_thr].sum()

        # EL Rate
        accept_all_el_rate = accept_all_el/accept_all_amt
        accept_thr_el_rate = accept_thr_el/accept_thr_amt

        # Implied LGD
        accept_all_lgd = (accept_all_el / (accept_all_mean_pd * accept_all_amt) if (accept_all_amt > 0 and accept_all_mean_pd > 0) else np.nan)
        accept_thr_lgd = (accept_thr_el / (accept_thr_mean_pd * accept_thr_amt) if (accept_thr_amt > 0 and accept_thr_mean_pd > 0) else np.nan)

        # Default Rate
        accept_all_dr  = y_true[accept_all].mean()
        accept_thr_dr  = y_true[accept_thr].mean()

        if verbose:

            print(f"\n{dataset_name}")
            print("-" * 62)
            print(f"{'Metric':<22}{'All':>18}{'Threshold':>18}")
            print("-" * 62)

            print(f"{'Acceptance Rate':<22}{accept_all_rate:>18.2%}{accept_thr_rate:>18.2%}")
            print(f"{'Mean PD':<22}{accept_all_mean_pd:>18.4f}{accept_thr_mean_pd:>18.4f}")
            print(f"{'Exposure':<22}{accept_all_amt:>18,.0f}{accept_thr_amt:>18,.0f}")
            print(f"{'Mean Exposure':<22}{accept_all_mean_amt:>18,.0f}{accept_thr_mean_amt:>18,.0f}")
            print(f"{'EL':<22}{accept_all_el:>18,.0f}{accept_thr_el:>18,.0f}")
            print(f"{'EL Rate':<22}{accept_all_el_rate:>18.2%}{accept_thr_el_rate:>18.2%}")
            print(f"{'Implied LGD':<22}{accept_all_lgd:>18.2%}{accept_thr_lgd:>18.2%}")
            print(f"{'Default Rate':<22}{accept_all_dr:>18.2%}{accept_thr_dr:>18.2%}")

        return

# function: my regression metrics

def my_regression_metrics(y_obs, y_pred, dataset_name="", verbose=True):
    
    # -------------------------
    # MAE (mean absolute error)
    # -------------------------
    mae = mean_absolute_error(y_obs, y_pred)

    # -------------------------
    # RMSE (root mean squared error)
    # -------------------------
    rmse = root_mean_squared_error(y_obs, y_pred)

    # -------------------------
    # CORRELATION (Pearson)
    # -------------------------
    corr = float(np.corrcoef(y_obs, y_pred)[0,1])

    # -------------------------
    # REGRESSION FIT
    # -------------------------    
    r2   = r2_score(y_obs, y_pred)

    # -------------------------
    # dictionary
    # -------------------------
    metrics = {
        "Dataset"     : dataset_name,
        "MAE"         : mae,
        "RMSE"        : rmse,
        "CORR"        : corr,
        "R2"          : r2,
    }

    # -------------------------
    # optional print
    # -------------------------
    if verbose:
        print("-" * 40)
        print(dataset_name)
        print("-" * 40)
        print(f"MAE         : {mae:.3f}")
        print(f"RMSE        : {rmse:.3f}")
        print(f"CORR        : {corr:.3f}")
        print(f"R2          : {r2:.3f}")
        print("-" * 40)

    return metrics

# function: my monte-carlo metrics

def my_monte_carlo_metrics(df, sim_dr, sim_losses, sim_losses_indiv, n_simulations, el_total, verbose=True):

    #    sim_dr           : simulated DR
    #    sim_losses       : per simulation portfolio loss
    #    sim_losses_indiv : per simulation individual losses

    # Monte-Carlo: validate

    print(40*'-')
    print('Monte Carlo validation')
    print(40*'-')
    val_summary = {
        "DR (mean) observed"  : df['default12'].mean(),
        "PD (mean) predicted" : df['PD'].mean(),
        "Simulated DR (mean)" : np.mean(sim_dr),
        "Simulated DR (std)"  : np.std(sim_dr),
        "Nr simulations"      : n_simulations,
    }
    for key, value in val_summary.items():
        if "Nr" in key:
            print(f"{key:<20}: {value:>12.0f}")
        else:
            print(f"{key:<20}: {value:>12,.4f}")        
    print(40*'-')

    # Monte-Carlo: summarize simulated losses

    print(40*'-')
    print("Portfolio Loss summary")
    print(40*'-')
    loss_summary = {
        "EAD"        : df["Amount"].sum(),
        "Loss (min)" : sim_losses.min(),
        "Loss (mean)": sim_losses.mean(),
        "Loss (std)" : sim_losses.std(),
        "Loss (max)" : sim_losses.max(),
        "VaR 95%"    : np.percentile(sim_losses, 95),
        "VaR 99%"    : np.percentile(sim_losses, 99),
        "VaR 99.9%"  : np.percentile(sim_losses, 99.9),
    }
    loss_summary["Economic Capital"] = (loss_summary["VaR 99.9%"] - loss_summary["Loss (mean)"])
    for key, value in loss_summary.items():
        print(f"{key:<20}: {value:>12,.0f}")
    print(40*'-')

    # Monte-Carlo: compare EL with simulated EL

    print(40*'-')
    print("Expected Loss comparison")
    print(40*'-')
    el_summary = {
        "EL (deterministic)" : el_total,
        "EL (MC-simulated)"  : sim_losses.mean(),        
    }
    el_summary["Difference"] = (el_summary["EL (deterministic)"] - el_summary["EL (MC-simulated)"])
    el_summary["Difference (rel)"] = el_summary["Difference"]/el_summary["EL (deterministic)"]*100
    for key, value in el_summary.items():
        if "(rel)" in key:
            print(f"{key:<20}: {value:>11.2f}%")
        else:
            print(f"{key:<20}: {value:>12,.0f}")
    print(40*'-')

    return val_summary, loss_summary, el_summary