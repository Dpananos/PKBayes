library(bayesplot)
library(here)
library(rstan)
suppressPackageStartupMessages(library(tidyverse))
library(tidybayes)

options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

`%notin%` <- Negate(`%in%`)


# --- Load in data and model ---

# Load Simulated data. Only want 100 subjects
data_location = here('data','simulated_data.csv')
d = read_csv(data_location) %>%
  filter(subjectids<=100)

obs_times = seq(0.5, 12, 0.5)

# Training
condition = d %>% 
  filter(times %in% obs_times)

# Testing
no_condition = d %>% 
  filter(times %notin% obs_times)

model_data = list(
  yobs = condition$Cobs,
  subjectids = condition$subjectids,
  n_subjects = length(unique(condition$subjectids)),
  times = condition$times,
  N = nrow(condition),
  #----------pred data---------
  Ntest = nrow(no_condition),
  test_ids = no_condition$subjectids,
  test_times = no_condition$times
)

saveRDS(model_data, here('data','simulated_data_dump.Rdump'))


#Load model for HMC and MAP
model_file = here('models','strong_model.stan')
model = stan_model(model_file)

# ---- Fit Model with HMC ----
#If this takes too long, thin the chains with a frequency of 3.
# Gives similar results, with only the out of sample error changing to within 1e-3
fit = sampling(model,
               model_data, 
               warmup=1000,
               iter=4000,
               chains = 12,
               # thin = 3,
               control = list(adapt_delta=0.8))


p = rstan::extract(fit)

mcmc_pred = apply(p$ypred, 2, mean)

# Make predictions and save to file
mcmc_pred = apply(p$ypred, 2, mean)
mcmc_low = apply(p$ypred, 2, function(x) quantile(x, 0.025))
mcmc_high = apply(p$ypred, 2, function(x) quantile(x, 0.975))

predictions = tibble(mcmc_pred, mcmc_low, mcmc_high)

data_location = here('data','mcmc_predictions.csv')
predictions %>% 
  rename(pred = mcmc_pred, low = mcmc_low, high = mcmc_high) %>% 
  bind_cols(no_condition) %>% 
  mutate(type='mcmc') %>% 
  write_csv(data_location)


# Save parameters for later

m = as.matrix(fit)
param_names = colnames(m)
mcmc_params = list(
  ke = m[, grepl('ke',param_names)],
  ka = m[, grepl('ka',param_names)],
  cl = m[, grepl('Cl',param_names)]
)

data_location = here('data','mcmc_parameter_draws.RDS')
saveRDS(mcmc_params, data_location)

