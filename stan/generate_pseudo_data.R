library(rstan)
library(tidyverse)


N_subjects = 250

sample_times = seq(0.5, 12, 0.25)
ids = 1:N_subjects
pairs = crossing(ids, sample_times) 

subjectids = pairs$ids
times = pairs$sample_times
N = length(times)

model_data = list(N = N, times = times, subjectids = subjectids, N_subjects = N_subjects)

model = stan_model('stan/generate_pseudo_data.stan')


fits = sampling(model, 
                data = model_data, 
                iter=1, 
                chains=1, 
                algorithm='Fixed_param',
                seed=19920908)


samples = rstan::extract(fits)


data = tibble(
  subjectids = subjectids,
  times = times,
  C = as.numeric(samples$C),
  Cobs = as.numeric(samples$Cobs),
) %>% 
  left_join(
    tibble(subjectids=unique(subjectids), 
           tmax = as.numeric(samples$tmax))
  )

data %>% 
  write_csv('data/simulated_data.csv')


