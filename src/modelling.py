import numpy as np
import pandas as pd
from scipy.stats import alpha
from src.metrics import calculate_monte_carlo_metrics
from IPython.display import display

# function: select n features using gb importance

def select_features_gb(y_trai, X_trai, n):

    # import
    import xgboost

    # initial model gb:
    model_gb = xgboost.XGBClassifier(random_state=888, max_depth=2, learning_rate=0.05)
    model_gb.fit(X_trai, y_trai)

    # feature importance
    importance = pd.Series(model_gb.feature_importances_, index=X_trai.columns)
    # sort
    feature_importance = importance.sort_values(ascending=False)
    # take first n
    feature_importance = feature_importance.head(n)
    # save names to list
    selected_features = feature_importance.index.tolist()

    return selected_features, feature_importance

# function: apply PD pipeline 
#           - calculate PD
#           - calculate EL       (optional)
#           - sensitivity on LGD (optional)

def apply_pipe_PD(df, pipeline, column_lgd=[], column_ead=[], calculate_el=False, sensitivity_lgd=False):

    df = df.copy()
    df["PD"]    = pipeline.predict_proba(df)[:,1]

    if calculate_el:
        # calculate expected loss using realised LGD 
        df["EL"]    = df["PD"] * df[column_ead] * df[column_lgd]

        if sensitivity_lgd:
            # calculate expected loss using LGD 10%, 15%, 25%, 90% 
            df["EL_10"] = df["PD"] * df[column_ead] * 0.10
            df["EL_15"] = df["PD"] * df[column_ead] * 0.15
            df["EL_25"] = df["PD"] * df[column_ead] * 0.25
            df["EL_90"] = df["PD"] * df[column_ead] * 0.90

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

def estimate_el(df, column_lgd, column_ead):

    df = df.copy()

    df["EL"]    = df["PD"] * df[column_ead] * df[column_lgd]

    return df

# function: fit LGD

def fit_lgd(lgd_dataset, method, one_value=0.15, one_variable=None, verbose=False):

    # import
    from sklearn.linear_model import LinearRegression
    import xgboost

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

        if verbose:
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
        model_X = lgd_dataset.drop(columns=["LossGivenDefault"]) # REVIEW LATER!
        # model y
        model_y = lgd_dataset["LossGivenDefault"]                                  
        # model
        model = LinearRegression()
        # fit
        lgd_model = {"method": method, "model": model.fit(model_X, model_y)}
        # coefficients
        coef = pd.Series(model.coef_, index=model_X.columns).sort_values(key=abs, ascending=False)
        if verbose:
            print(coef)

    if method == "gradient_boosting_regression":

        # model X
        model_X = lgd_dataset.drop(columns=["LossGivenDefault", "num__default12"], errors="ignore") # REVIEW LATER!
        # model y
        model_y = lgd_dataset["LossGivenDefault"]                             
        # model
        model = xgboost.XGBRegressor()
        # fit
        lgd_model = {"method": method, "model": model.fit(model_X, model_y)}
        # importances
        impo = pd.Series(model.feature_importances_, index=model_X.columns).sort_values(ascending=False)
        if verbose:
            print(impo)

    if method == "stochastic":

        # model y
        model_y = lgd_dataset["LossGivenDefault"]
        # fit 
        lgd_model = {"method": method, "model": fit_stochastic_lgd(model_y, verbose)}

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

        model_X = df.drop(columns=["LossGivenDefault", "num__default12"], errors="ignore") # REVIEW LATER!
        df["LGD"] = lgd_model["model"].predict(model_X)
        df["LGD"] = df["LGD"].clip(0, 1)

    if method == "gradient_boosting_regression":

        model_X = df.drop(columns=["LossGivenDefault", "num__default12"], errors="ignore") # REVIEW LATER!
        df["LGD"] = lgd_model["model"].predict(model_X)
        df["LGD"] = df["LGD"].clip(0, 1)

    if method == "stochastic":

        model_X = df.drop(columns=["LossGivenDefault", "num__default12"], errors="ignore") # REVIEW LATER!
        df["LGD"] = transform_stochastic_lgd(model_X, lgd_model)
        df["LGD"] = df["LGD"].clip(0, 1)

    return df

# function: fit stochastic LGD

def fit_stochastic_lgd(model_y, verbose=False):

    # import
    from scipy.stats import beta

    # Zero-inflation
    pi_zero = (model_y == 0).mean()

    # Positive LGDs
    lgd_positive = model_y[model_y > 0]

    # Fit Beta distribution to positive LGDs
    alpha, beta_param, loc, scale = beta.fit(
        lgd_positive,
        floc=0,
        fscale=1
    )

    if verbose:
        empirical_mean = lgd_positive.mean()
        beta_mean = beta.mean(alpha, beta_param)

        empirical_variance = lgd_positive.var()
        beta_variance = beta.var(alpha, beta_param)

        quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
        empirical_q = lgd_positive.quantile(quantiles)
        beta_q = beta.ppf(
            quantiles,
            alpha,
            beta_param
        )

        print(f"Zero LGD probability:     {pi_zero:.4f}")
        print(f"Positive LGD probability: {1 - pi_zero:.4f}")

        print("\nPositive LGD distribution:")
        print(f"  Empirical mean:     {empirical_mean:.4f}")
        print(f"  Beta mean:          {beta_mean:.4f}")
        print(f"  Empirical variance: {empirical_variance:.6f}")
        print(f"  Beta variance:      {beta_variance:.6f}")

        print("\nQuantile | Empirical | Beta")
        for q, e, f in zip(quantiles, empirical_q, beta_q):
            print(f"{q:.2f} | {e:.4f} | {f:.4f}")

    return {
    "pi_zero": pi_zero,
    "alpha":   alpha,
    "beta":    beta_param,
    }

# function: transform stochastic LGD

def transform_stochastic_lgd(df, lgd_model):

    # import
    from scipy.stats import beta

    # random numbers
    u = np.random.random(len(df))

    # set all to -> 0 first
    lgd = np.zeros(len(df))

    # read beta model parameters
    pi_zero    = lgd_model["model"]["pi_zero"]
    alpha      = lgd_model["model"]["alpha"]
    beta_param = lgd_model["model"]["beta"]

    # distinguish lgd positive part
    positive = u >= pi_zero

    # transform lgd positive part
    lgd[positive] = beta.rvs(alpha, beta_param, size=positive.sum())

    return lgd

# function: estimate capital

def estimate_capital(df, method, column_lgd, column_def, column_ead, allocate= False, verbose=False):

    df = df.copy()

    # calculate EL
    df["EL"] = df["PD"] * df[column_ead] * df[column_lgd]

    if verbose:
        # portfolio total exposure
        print('Portfolio EAD : ', df[column_ead].sum())
        # portfolio EL (deterministic)
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
            loss = (sim_defaults * df[column_ead] * df[column_lgd])

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
            print('sim_dr.shape=', sim_dr.shape) # portfolio default rate in simulation i
            print('sim_losses.shape=', sim_losses.shape) # portfolio loss in simulation i
            print('sim_losses_indiv.shape=', sim_losses_indiv.shape) # loss of loan j in simulation i
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

        val_summary, loss_summary, el_summary = calculate_monte_carlo_metrics(df, 
                                                                    sim_dr, 
                                                                    sim_losses, 
                                                                    sim_losses_indiv, 
                                                                    n_simulations, 
                                                                    df["EL"].sum(),
                                                                    column_def, 
                                                                    column_ead, 
                                                                    verbose=verbose)

        # save output
        
        mc_el  = el_summary["EL (MC-simulated)"]
        mc_cap = loss_summary["Economic Capital"]

        # Monte-Carlo: loss distribution
        # removed 

        # Monte-Carlo: allocate 

        if allocate:

            # define tail

            tail        = sim_losses >= loss_summary["VaR 99.9%"]
            tail_losses = sim_losses_indiv[tail]

            if verbose:
    
                print('- shapes:')
                print('tail_losses.shape=', tail_losses.shape)
                print('-')

            df["CAP"]   = tail_losses.mean(axis=0)

            if verbose:

                print('- consistency check:')
                print('df["CAP"].sum()=', df["CAP"].sum())
                print('tail_losses.sum(axis=1).mean()=', tail_losses.sum(axis=1).mean())
                print('-')

            # average contribution in the tail (one value per loan)

            tail_contrib = tail_losses.mean(axis=0)

            # allocate portfolio economic capital

            capital_alloc = (tail_contrib / tail_contrib.sum()) * loss_summary["Economic Capital"]

            # assign to each loan

            df["CAP"] = capital_alloc

            if verbose:

                print('- consistency check:')
                print('df["CAP"].sum()=', df["CAP"].sum())
                print('loss_summary["Economic Capital"]=', loss_summary["Economic Capital"])
                print('-')

    # end of estimation

    return df, loss_summary, el_summary, sim_losses
