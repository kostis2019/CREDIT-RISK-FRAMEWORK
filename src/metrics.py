import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    brier_score_loss
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