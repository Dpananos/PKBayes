library(rstan)
library(bayesplot)
library(tidyverse)
library(tidybayes)
options(mc.cores = parallel::detectCores())

model = stan_model('~/Documents/PhD Code/PKBayes/stan/model.stan')

d = read_csv('~/Documents/PhD Code/PKBayes/data/apixiban_regression_data.csv')

model_data = list(
  yobs = d$Concentration_scaled,
  subjectids = as.integer(factor(d$Subject)),
  n_subjects = 36,
  times = d$Time,
  N = nrow(d)
)


fit = sampling(model, model_data, chains = 12, warmup = 2000, iter = 4000)

p = rstan::extract(fit)

