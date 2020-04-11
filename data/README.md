# Codebook

### apixiban_regression_data.csv

Concentration data used to fit our apixaban model.  This is not generated from any script.

* `Time`: Time (in hours) when observation was made
* `Subject`:  Participant ID Number
* `Concentration`: Concentration observed in ng/ml
* `Group`: If subject is control or has non-alcoholic fatty liver disease
* `Sex`: Male/Female
* `Age`: In years
* `Weight`: In kilograms
* `Creatinine`: In micromolar.  Measure of kidney function 
* `BMI`: Body Mass Index

### simulated_data.RDS

This dataset is created in `analysis/01_fit_apixaban_model.R`.  It is the result of sampling from `models/generate_pseudodata.stan`.  It contains 250 pseudopatients, their concentration observations, and their latent concentrations, and their pharmacokinetic parameters.

### simulated_data.csv

This dataset is created in `analysis/01_fit_apixaban_model.R` and contains some of the data from `simulated_data.RDS` in tidyformat.  It exists for convienience and to pass to Stan models.

* `subjectids`: Psuedopatient ID Number
* `times`: Time of observation (in hours)
* `C`: Latent concentration at corresponding `times`.  Noiseless concentration
* `Cobs`:  Observation with noise
* `tmax`: Pseudopatient parameter for max concentration
* `cl`: Pseudopatient parameter for clearance rate
* `ke`: Pseudopatient parameter for elimination rate constant
* `ka`: Pseudopatient parameter for absorption rate constant
* `alpha`: Pseudopatient parameter for ratio of `ke` and `ka`.

### map_predictions.csv

This dataset is created in `analysis/02_Run_MAP_Fit.R`. It contains the predctions and credible intervals for pseudopatient latent concentration.

* `map_pred`: Predicted concentration for pseudopatients.  Predicted concentration is computed using expectations.
* `map_low`:  Lower equal tailed posterior interval limit.
* `map_high`: Upper equal tailed posterior interval limit.

### map_parameter_draws.RDS

This dataset is created in `analysis/02_Run_MAP_Fit.R` and contains a list of matrices.  The matrices in these lists are 10000x100 matrices representing the posterior samples for the pseudopatient pharmacokinetic parameters `cl`, `ke`, and `ka`.  Entry `[i,j]` in these matrices represents the `ith` sample for the `jth` pseudopatient.

### mcmc_predictions.csv

This dataset is created in `analysis/03_Run_HMC_Fit.R`. It contains the predctions and credible intervals for pseudopatient latent concentration.

* `mcmc_pred`: Predicted concentration for pseudopatients.  Predicted concentration is computed using expectations.
* `mcmc_low`:  Lower equal tailed posterior interval limit.
* `mcmc_high`: Upper equal tailed posterior interval limit.

### mcmc_parameter_draws.RDS

This dataset is created in `analysis/03_Run_HMC_Fit.R` and contains a list of matrices.  The matrices in these lists are 30000x100 matrices representing the posterior samples for the pseudopatient pharmacokinetic parameters `cl`, `ke`, and `ka`.  Entry `[i,j]` in these matrices represents the `ith` sample for the `jth` pseudopatient.


### experiment_1_doses.csv

This dataset is created in `analysis/04_12_Hour_Calibration.R`.  It contains estimated doses to achieve an indicated risk for our risk-at-12-hours experiment.

* `patient`:  Pseudopatient ID Number
* `p`: Desired risk level
* `mcmc_estimated_dose`: Dose estimated from HMC posterior
* `map_estimated_dose`: Dose estimated from MAP posterior



### experiment_2_doses.csv

This dataset is created in `analysis/05_Cmax_Calibration.R`.  It contains estimated doses to achieve an indicated risk for our risk-at-tmax experiment.

* `patient`:  Pseudopatient ID Number
* `p`: Desired risk level
* `mcmc_estimated_dose`: Dose estimated from HMC posterior
* `map_estimated_dose`: Dose estimated from MAP posterior

### table_2_errors.csv

Out of sample expexted error (and standard deviations) for each model.  This data populates table 2 in the paper.