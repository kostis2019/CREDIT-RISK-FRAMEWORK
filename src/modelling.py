import numpy as np
import pandas as pd
from src.feature_engineering import create_target_def12
from src.metrics import my_monte_carlo_metrics
from src.preprocessing import my_input_load
from src.plots import plot_loss_distribution
from src.utils import display_table
from sklearn.linear_model import LinearRegression
import xgboost

# function: apply PD pipeline 
#           - calculate PD
#           - calculate EL       (optional)
#           - sensitivity on LGD (optional)

def apply_pipe(df, pipeline, calculate_el=False, sensitivity_lgd=False):

    df = df.copy()
    df["PD"]    = pipeline.predict_proba(df)[:,1]

    if calculate_el:
        # calculate expected loss using realised LGD 
        df["EL"]    = df["PD"] * df["Amount"] * df["LossGivenDefault"]

        if sensitivity_lgd:
            # calculate expected loss using LGD 10%, 15%, 25%, 90% 
            df["EL_10"] = df["PD"] * df["Amount"] * 0.10
            df["EL_15"] = df["PD"] * df["Amount"] * 0.15
            df["EL_25"] = df["PD"] * df["Amount"] * 0.25
            df["EL_90"] = df["PD"] * df["Amount"] * 0.90

    return df

# function: apply LGD pipeline

def apply_pipe_LGD(df, pipeline):

    df = df.copy()

    # preprocess
    X = pipeline["preprocessor"].transform(df)

    # estimate LGD
    X = transform_lgd(X, pipeline["model"])

    # append
    df["LGD"] = X["LGD"]

    return df

# function: estimate EL 

def estimate_el(df, column_lgd):

    df = df.copy()

    df["EL"]    = df["PD"] * df["Amount"] * df[column_lgd]

    return df

# function: estimate LGD (replaced by fit_lgd, transform_lgd)

def estimate_lgd_v0(df, lgd_dataset, method, one_value=0.15, one_variable=None, verbose=False):

    df = df.copy()

    ###### ESTIMATE ######

    print('LGD estimation method: ', method)
    print('Feature: ', one_variable)

    if method == 'one_value':

        value    = one_value

    if method == 'historical_avg':

        value    = lgd_dataset['LossGivenDefault'].mean()
        
    if method == "univariate":

        # EVALUATION DF
        df_eval = pd.DataFrame({
        "feature": lgd_dataset[one_variable],
        "lgd"    : lgd_dataset["LossGivenDefault"],
        })

        # Categorical OR Numerical?
        if df_eval["feature"].nunique() > 20:

            numerical = True

        else:

            numerical = False

        # Numerical variable → bin
        if numerical:

            bins = pd.qcut(df_eval["feature"], q=10, duplicates="drop", retbins=True)[1]
            df_eval["group"] = pd.cut(df_eval["feature"], bins=bins, precision=2, include_lowest=True)

        # Categorical variable
        else:

            df_eval["group"] = df_eval["feature"]

        # AVERAGE LGD AND RECOVERY RATE
        lgd_table = (
        df_eval
        .groupby("group", observed=False)
        .agg(
        count=("lgd", "count"),
        lgd=("lgd", "mean"),
        std=("lgd", "std"),
        recovery_rate=("lgd", lambda x: (x == 0).mean()),
        positive_lgd=("lgd", lambda x: x[x > 0].mean())
        ))
        if verbose:
            display(lgd_table)

        # MAP
        lgd_mapping = lgd_table["lgd"].to_dict()
        if verbose:
            display(lgd_mapping)

    if method == 'linear_regression':
        
        print('in progress')

    if method == 'gradient_boosting_regression':
        
        print('in progress')

    ###### APPLY ######

    if method == 'one_value':

        df["LGD"]    = value

    if method == 'historical_avg':

        df["LGD"]    = value

    if method == "univariate":

        if numerical:

            df["group"] = pd.cut(df[one_variable], bins=bins, include_lowest=True)
            df["LGD"] = df["group"].map(lgd_mapping)

        else:

            df["LGD"] = df[one_variable].map(lgd_mapping)

        # unseen categories
        df["LGD"] = df["LGD"].fillna(lgd_table["lgd"].mean())

    return df

# function: fit LGD

def fit_lgd(lgd_dataset, method, one_value=0.15, one_variable=None, verbose=False):

    print("LGD estimation method:", method)

    lgd_model = {
        "method": method,
        "variable": one_variable,
    }

    if method == "one_value":

        lgd_model["value"] = one_value

    if method == "historical_avg":

        lgd_model["value"] = lgd_dataset["LossGivenDefault"].mean()

    if method == "univariate":

        print("Feature:", one_variable)

        # EVALUATION DF
        df_eval = pd.DataFrame({
            "feature": lgd_dataset[one_variable],
            "lgd": lgd_dataset["LossGivenDefault"],
        })

        # Categorical OR Numerical?
        numerical = lgd_dataset[one_variable].nunique() > 20

        lgd_model["numerical"] = numerical

        if numerical:

            bins = pd.qcut(
                df_eval["feature"],
                q=10,
                duplicates="drop",
                retbins=True
            )[1]

            df_eval["group"] = pd.cut(
                df_eval["feature"],
                bins=bins,
                precision=2,
                include_lowest=True
            )

            lgd_model["bins"] = bins

        else:

            df_eval["group"] = df_eval["feature"]

        lgd_table = (
            df_eval
            .groupby("group", observed=False)
            .agg(
                count=("lgd", "count"),
                lgd=("lgd", "mean"),
                std=("lgd", "std"),
                recovery_rate=("lgd", lambda x: (x == 0).mean()),
                positive_lgd=("lgd", lambda x: x[x > 0].mean()),
            )
        )

        if verbose:
            display(lgd_table)

        lgd_model["mapping"] = lgd_table["lgd"].to_dict()
        lgd_model["fallback"] = lgd_table["lgd"].mean()

    if method == "linear_regression":

        # model X
        model_X = lgd_dataset.drop(columns=["LossGivenDefault", "num__default12"])
        # model y
        model_y = lgd_dataset["LossGivenDefault"]
        # model
        model = LinearRegression()
        # fit
        lgd_model = {"method": method, "model": model.fit(model_X, model_y)}
        # coefficients
        coef = pd.Series(model.coef_, index=model_X.columns).sort_values(key=abs, ascending=False)
        print(coef)

    if method == "gradient_boosting_regression":

        # model X
        model_X = lgd_dataset.drop(columns=["LossGivenDefault", "num__default12"])
        # model y
        model_y = lgd_dataset["LossGivenDefault"]
        # model
        model = xgboost.XGBRegressor()
        # fit
        lgd_model = {"method": method, "model": model.fit(model_X, model_y)}
        # importances
        impo = pd.Series(model.feature_importances_, index=model_X.columns).sort_values(ascending=False)
        print(impo)

    return lgd_model

# function: transform LGD

def transform_lgd(df, lgd_model):

    df = df.copy()

    method = lgd_model["method"]

    if method == "one_value":

        df["LGD"] = lgd_model["value"]

    if method == "historical_avg":

        df["LGD"] = lgd_model["value"]

    if method == "univariate":

        if lgd_model["numerical"]:

            df["group"] = pd.cut(
                df[lgd_model["variable"]],
                bins=lgd_model["bins"],
                include_lowest=True
            )

            df["LGD"] = df["group"].map(lgd_model["mapping"])

        else:

            df["LGD"] = df[lgd_model["variable"]].map(
                lgd_model["mapping"]
            )

        df["LGD"] = df["LGD"].fillna(
            lgd_model["fallback"]
        )

    if method == "linear_regression":

        model_X = df.drop(columns=["LossGivenDefault", "num__default12"], errors="ignore")
        df["LGD"] = lgd_model["model"].predict(model_X)
        df["LGD"] = df["LGD"].clip(0, 1)

    if method == "gradient_boosting_regression":

        model_X = df.drop(columns=["LossGivenDefault", "num__default12"], errors="ignore")
        df["LGD"] = lgd_model["model"].predict(model_X)
        df["LGD"] = df["LGD"].clip(0, 1)

    return df

# function: estimate capital

def estimate_capital(df, method, column_lgd, allocate= False, verbose=False):

    df = df.copy()

    # portfolio total exposure

    print('Portfolio EAD : ', df["Amount"].sum())

    # portfolio EL (deterministic)

    df["EL"] = df["PD"] * df["Amount"] * df[column_lgd]
    print('Portfolio EL  : ', df["EL"].sum())

    # function outputs

    mc_el  = None
    mc_cap = None

    # reference line: "simple" 2 * expected loss

    reference_2    =  2 * df["EL"].sum()

    # reference line: "stress" 5 * expected loss
      
    reference_5    =  5 * df["EL"].sum()

    # method "monte-carlo"

    if method == "monte-carlo":
        
        # Monte-Carlo: initialise

        rng = np.random.default_rng(seed=42)
        n_simulations = 10000

        sim_dr           = []
        sim_losses       = []
        sim_losses_indiv = [] 

        # Monte-Carlo: simulate

        for i in range(n_simulations):
    
            # simulate defaults
            sim_defaults = rng.binomial(n=1, p=df["PD"])
            sim_dr.append(sim_defaults.mean())

            # simulate losses
            loss = (sim_defaults * df["Amount"] * df[column_lgd])

            # portfolio loss for simulation i
            loss_total = loss.sum()
            sim_losses.append(loss_total)

            # individual losses for simulation i
            loss_indiv = loss
            sim_losses_indiv.append(loss_indiv)

        sim_dr           = np.array(sim_dr)
        sim_losses       = np.array(sim_losses)
        sim_losses_indiv = np.array(sim_losses_indiv)

        if verbose:

            # check shapes
        
            print('- shapes:')
            print(sim_dr.shape)           # sim_dr[i]              : portfolio default rate in simulation i
            print(sim_losses.shape)       # sim_losses[i]          : portfolio loss in simulation i
            print(sim_losses_indiv.shape) # sim_losses_indiv[i, j] : loss of loan j in simulation i
            print('-')

            # consistency check

            print('- consistency check:')
            print(np.allclose(sim_losses, sim_losses_indiv.sum(axis=1)))
            print('-')

        # Monte-Carlo: output

        sim_dr           = sim_dr            # simulated DR
        sim_losses       = sim_losses        # per simulation: portfolio loss
        sim_losses_indiv = sim_losses_indiv  # per simulation: individual losses

        # Monte-Carlo: metrics

        val_summary, loss_summary, el_summary = my_monte_carlo_metrics(df, 
                                                                    sim_dr, 
                                                                    sim_losses, 
                                                                    sim_losses_indiv, 
                                                                    n_simulations, 
                                                                    el_total=df["EL"].sum(), 
                                                                    verbose=verbose)

        # save output
        
        mc_el  = el_summary["EL (MC-simulated)"]
        mc_cap = loss_summary["Economic Capital"]

        # Monte-Carlo: loss distribution

        if verbose:

            plot_loss_distribution(sim_losses, df["Amount"].sum())

        # Monte-Carlo: allocate 

        if allocate:

            # define tail

            tail        = sim_losses >= loss_summary["VaR 99.9%"]
            tail_losses = sim_losses_indiv[tail]

            if verbose:
    
                print('- shapes:')
                print(tail_losses.shape)
                print('-')

            df["CAP"]   = tail_losses.mean(axis=0)

            if verbose:

                print('- consistency check:')
                print(df["CAP"].sum())
                print(tail_losses.sum(axis=1).mean())
                print('-')

            # average contribution in the tail (one value per loan)

            tail_contrib = tail_losses.mean(axis=0)

            # allocate portfolio economic capital

            capital_alloc = (tail_contrib / tail_contrib.sum()) * loss_summary["Economic Capital"]

            # assign to each loan

            df["CAP"] = capital_alloc

            if verbose:

                print('- consistency check:')
                print(df["CAP"].sum())
                print(loss_summary["Economic Capital"])
                print('-')

    # end of estimation
    return df, mc_el, mc_cap