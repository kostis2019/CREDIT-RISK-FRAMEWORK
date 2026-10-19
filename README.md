# Credit Risk Modelling Framework

A modular Python framework for developing, validating, and applying credit-risk models from Probability of Default (PD) through portfolio risk and credit decisioning. It started from a concrete application-time credit-risk modelling problem and evolved into a reusable framework for the broader credit-risk lifecycle.

## Overview

The framework covers:

⚫ PD model development and temporal validation<br>
⚫ PD calibration and recalibration<br>
⚫ LGD estimation and EAD integration<br>
⚫ Expected Loss (EL) analysis<br>
⚫ Monte Carlo portfolio loss simulation and Economic Capital<br>
⚫ Credit-policy and PD-threshold analysis<br>
⚫ Model diagnostics, explainability and fairness<br>
⚫ Input drift monitoring<br>
⚫ Automated testing and CI<br>

<p align="center">
  <img src="figures/framework_summary_new.png" width="500">
</p>

## Recommended Walkthrough

For a first look at the framework:

1. **[The First Case-Study](notebooks/00_first_case_study.ipynb)** — the initial business problem and the first modelling approach.
2. **[The PD Pipeline](notebooks/02_PD_pipeline.ipynb)** — the generalized PD modelling workflow.
3. **[Portfolio Risk Report](slides/reports/First_Report.pdf)** — portfolio overview, risk, and capital analysis.

## Future Extensions

Planned extensions include:

⚪ Stochastic LGD modelling<br>
⚪ Dedicated EAD modelling<br>
⚪ Inverse portfolio-risk analysis<br>
⚪ Configuration driven execution-retraining<br>
⚪ Model health dashboard<br>