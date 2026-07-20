import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone #FlexibleCalibratedModel
from sklearn.calibration import CalibratedClassifierCV         #FlexibleCalibratedModel
from sklearn.linear_model import LogisticRegression            #PDCalibrator
from sklearn.isotonic import IsotonicRegression                #PDCalibrator

# function: intercept recalibration

def logit(p):
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return np.log(p / (1 - p))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def intercept_recalibration(y_true, pd_pred):
    """
    y_true  : array-like (0/1)
    pd_pred : predicted probabilities (PDs)
    
    returns:
        pd_calibrated : recalibrated PDs
        delta         : intercept shift applied
    """
    
    # Mean observed default rate
    y_bar = np.mean(y_true)
    
    # Mean predicted PD
    p_bar = np.mean(pd_pred)
    
    # Compute intercept shift
    delta = np.log(y_bar / (1 - y_bar)) - np.log(p_bar / (1 - p_bar))
    
    # Apply shift in logit space
    pd_calibrated = sigmoid(logit(pd_pred) + delta)
    
    return pd_calibrated, delta

# class: flexible calibration (calibration inside pipeline)

class FlexibleCalibratedModel(BaseEstimator, ClassifierMixin):
    
    def __init__(self, base_model, calibration=None, shift=0.0):
        self.base_model = base_model
        self.calibration = calibration  # None, "sigmoid", "isotonic", "shift"
        self.shift = shift
    
    # def fit(self, X, y):
    #     self.model_ = clone(self.base_model)
    #     self.model_.fit(X, y)
        
    #     if self.calibration in ["sigmoid", "isotonic"]:
    #         self.calibrator_ = CalibratedClassifierCV(
    #             self.model_, method=self.calibration, cv=3
    #         )
    #         self.calibrator_.fit(X, y)
        
    #     return self
    
    def fit(self, X, y):
        self.model_ = clone(self.base_model)
        self.model_.fit(X, y)

        # Expose common fitted attributes
        if hasattr(self.model_, "coef_"):
            self.coef_ = self.model_.coef_

        if hasattr(self.model_, "intercept_"):
            self.intercept_ = self.model_.intercept_

        if hasattr(self.model_, "feature_names_in_"):
            self.feature_names_in_ = self.model_.feature_names_in_

        if self.calibration in ["sigmoid", "isotonic"]:
            self.calibrator_ = CalibratedClassifierCV(
                self.model_, method=self.calibration, cv=3
            )
            self.calibrator_.fit(X, y)

        return self

    def predict_proba(self, X):
        if self.calibration in ["sigmoid", "isotonic"]:
            return self.calibrator_.predict_proba(X)
        
        proba = self.model_.predict_proba(X)
        
        if self.calibration == "shift":
            p = proba[:, 1]
            logit = np.log(p / (1 - p))
            logit_shifted = logit + self.shift
            p_new = 1 / (1 + np.exp(-logit_shifted))
            return np.vstack([1 - p_new, p_new]).T
        
        return proba

# class: calibrator (independent calibration)

class PDCalibrator:

    def __init__(self, method="none", shift=0.0):

        self.method = method.lower()
        self.shift = shift

        self.fitted_ = False
        self.model_ = None

    def fit(self, predictions, y_true):

        pd_raw = predictions["PD"].values

        if self.method == "none":

            self.fitted_ = True

        elif self.method == "shift":

            self.fitted_ = True

        elif self.method == "platt":

            self.model_ = LogisticRegression()

            self.model_.fit(
                pd_raw.reshape(-1, 1),
                y_true
            )

            self.fitted_ = True

        elif self.method == "isotonic":

            self.model_ = IsotonicRegression(
                out_of_bounds="clip"
            )

            self.model_.fit(
                pd_raw,
                y_true
            )

            self.fitted_ = True

        else:

            raise ValueError(
                f"Unknown method: {self.method}"
            )

        return self

    def transform(self, predictions):

        if not self.fitted_:

            raise ValueError(
                "Calibrator must be fitted first."
            )

        df = predictions.copy()

        df["PD_raw"] = df["PD"]

        pd_raw = df["PD_raw"].values

        if self.method == "none":

            df["PD"] = pd_raw

        elif self.method == "shift":

            logit = np.log(
                pd_raw / (1 - pd_raw)
            )

            logit_shifted = (
                logit + self.shift
            )

            df["PD"] = (
                1 / (1 + np.exp(-logit_shifted))
            )

        elif self.method == "platt":

            df["PD"] = (
                self.model_
                .predict_proba(
                    pd_raw.reshape(-1, 1)
                )[:, 1]
            )

        elif self.method == "isotonic":

            df["PD"] = (
                self.model_
                .predict(pd_raw)
            )

        return df

    def fit_transform(
        self,
        predictions,
        y_true
    ):

        self.fit(
            predictions,
            y_true
        )

        return self.transform(
            predictions
        )