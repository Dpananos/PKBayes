# Models


### model.stan

We fit this model to `data/apixiban_regression_data.csv`.  See figure 2 in our paper for model exposition.  This model is used in `analysis/01_fit_apixaban_model.R`.

### prior_predictive_check.stan

This is a Stan model to sample from the prior for `model.stan`.  The prior in `model.stan` is hard coded, which makes iterating over prior configurations difficult.  This stand alone Stan file parameterizes the priors for ease of prior exploration. This code is used in `analysis/01_fit_apixaban_model.R`.

### generate_pseudo_data.stan

This file uses the summarized posterior from `model.stan` and uses it to generate pseudopatients.  This code is used in `analysis/01_fit_apixaban_model.R`.

### strong_model.stan

This model is the result of summarizing the posterior from `model.stan` and using those summaries are priors.  The model is "strong" because the priors are informed from the `data/apixiban_regression_data.csv`.  This model is used in `analysis/02_Run_MAP_Fit.R` and `analysis/03_Run_HMC_Fit.R`.
