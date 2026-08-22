### Credit Risk Modelling Framework

#### Introduction

This project is a modular Python framework for developing, calibrating, validating, and analysing application-time Probability of Default (PD) and Loss Given Default (LGD) models for unsecured consumer lending. It implements an end-to-end credit risk workflow, from data preprocessing and feature engineering to model calibration, portfolio decisioning, performance evaluation, and expected loss analysis.

The framework has been designed with reusability in mind. Instead of isolated notebooks, the core functionality is organised into reusable Python modules, making it easy to train new models, compare modelling strategies, evaluate different decision thresholds, and extend the framework with additional monitoring and governance capabilities.

The project is inspired by real-world credit risk modelling workflows used in financial institutions and aims to bridge the gap between exploratory data science notebooks and a reusable production-style machine learning framework.

<p align="center">
  <img src="figures/framework_summary_new.png" width="600">
</p>

#### The framework answers questions such as:

⚫ How can a robust application-time Probability of Default (PD) model be developed from historical lending data?<br>
⚫ Which preprocessing and feature engineering steps are required before modelling?<br>
⚫ How do Logistic Regression and Gradient Boosting compare for application-time PD modelling?<br>
⚫ How well calibrated are the predicted probabilities?<br>
⚫ Which calibration technique provides the best probability estimates?<br>
⚫ How can an optimal PD acceptance threshold be selected?<br>
⚫ What is the impact of different approval thresholds on Approval Rate, Default Rate and Expected Loss?<br>
⚫ How can application-time Loss Given Default (LGD) be estimated using empirical and modelling approaches?<br>
⚫ How can portfolio Expected Loss (EL) be quantified and visualized?<br>
⚫ How can portfolio loss distributions be estimated using Monte Carlo simulation?<br>
⚫ How can Value-at-Risk (VaR) and Economic Capital be estimated from simulated portfolio losses?<br>
⚫ How can Monte Carlo estimates be validated against deterministic Expected Loss?<br>
🟢 How can portfolio Economic Capital be allocated back to individual exposures?<br>
🟢 How can Population Stability Index (PSI) and feature drift be monitored?<br>
🟢 How can model governance and monitoring dashboards support retraining decisions?<br>
🟢 How can application-time LGD modelling be further improved?<br>
⚪ How can correlated defaults be incorporated into the Monte Carlo simulation?<br>
⚪ How can macroeconomic scenarios and stress testing be incorporated?<br>
⚪ How can the framework be executed through configuration files?<br>
⚪ How can the framework be deployed as an API?<br>

**Status:** ⚫ Implemented | 🟢 In Progress | ⚪ Planned