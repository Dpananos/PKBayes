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
d = here('data','simulated_data.csv') %>% 
  read_csv() %>%
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


# ---- Fit Model with MAP ----

maps = optimizing(
  model,
  data = model_data,
  verbose = TRUE,
  algorithm = 'LBFGS',
  as_vector = TRUE,
  hessian = TRUE,
  tol_obj=1e-10,
  iter=10000,
  draws = 10000,
  seed = 19920908
)

H = maps$hessian
S = MASS::ginv(-H)
dimnames(S) = dimnames(H)
theta_tilde = maps$theta_tilde

# Save predictions
ypred_cols = theta_tilde[, grepl('ypred', colnames(theta_tilde))]
map_pred = apply(ypred_cols, 2, mean)
map_low = apply(ypred_cols, 2, function(x) quantile(x, 0.025))
map_high = apply(ypred_cols, 2, function(x) quantile(x, 0.975))
predictions = tibble(map_pred, map_low, map_high)

data_location = here('data', 'map_predictions.csv')
predictions %>% 
  rename(pred = map_pred, low = map_low, high = map_high) %>% 
  bind_cols(no_condition) %>% 
  mutate(type='map') %>% 
  write_csv(data_location)


# Save parameters
param_draws = list(
  ke = theta_tilde[,grep('ke', colnames(theta_tilde))],
  ka = theta_tilde[,grep('ka', colnames(theta_tilde))],
  cl = theta_tilde[,grep('Cl', colnames(theta_tilde))]
)

data_location = here('data','map_parameter_draws.RDS')
saveRDS(param_draws, data_location)
