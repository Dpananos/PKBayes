library(here)
library(Metrics)
library(patchwork)
library(rstan)
library(tidybayes)
suppressPackageStartupMessages(library(tidyverse))
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())
theme_set(theme_minimal())

# ---- Prior Predictive Checks ----

# Load the prior predictive check as a model in Stan
prior_model <- here('models', 'prior_predictive_check.stan') %>% 
               stan_model()

# Times at which we evaluate the concentration function.
# We will be passing this to the prior predictive model and generate observations at these times
ts <- seq(0.5, 12, 0.1)

# Parameters for the prior.
# It is easier to alter priors when the prior hyperparameters are passed as data
# E.g. MU_CL_MEAN is the mean for the prior on the parameter mu_CL
prior_params <- list(
  N = length(ts),
  t = ts,
  MU_CL_MEAN = log(log(3.3)),
  MU_CL_SIGMA = 0.25,
  S_CL_A = 15,
  S_CL_B = 100,
  MU_T_MEAN = log(3.3),
  MU_T_SIGMA = 0.25,
  S_T_A = 10,
  S_T_B = 100,
  PHI_A = 20,
  PHI_B = 20,
  KAPPA_A = 20,
  KAPPA_B = 20,
  SIGMA_MEAN = log(0.1),
  SIGMA_SIGMA = 0.1
)

#Sample from the prior now
# Sample from the prior
prior_samples <- sampling( prior_model, data = prior_params, algorithm = 'Fixed_param', chains = 1)

# Plot the prior
units <- function(x) {
  glue::glue('{x} ng/ml')
}



# Plot the prior now.  One curve per sample.
# 250 samples from the prior will be plotted (see n argument in spread_draws)
# COncentrations from the model are in units mg/L
# Multiply by 1000 to get ng/ml, a more reasonable unit for pharmacokinetics
sampled_times <- tibble(i = seq_along(ts), t = ts)
prior_plot_data <- prior_samples %>%
  spread_draws(y[i], n = 250, seed = 1) %>%
  left_join(sampled_times)

prior_plot = prior_plot_data %>% 
  ggplot(aes(t, 1000 * y, group = .draw)) +
  geom_line(alpha = 0.2) +
  labs(x = 'Hours Post Dose',
       y = 'Concentration (ng/ml)',
       subtitle = 'Plausible Concentrations\nBefore Seeing Data') +
  ylim(0, 250) +
  theme(aspect.ratio = 1)


# ---- Fit Model ----

model <- here('models', 'model.stan') %>%  
         stan_model()

# Read in the data for the regresion
# Create a new column, i, so I can join it to the result from tidybayes::spread_draws later
d = here('data', 'apixaban_regression_data.csv') %>%  
  read_csv() %>%
  mutate(i = seq_along(Concentration)) 

# Prepare data to be sent to Stan
model_data <- list(
  yobs = d$Concentration_scaled,
  subjectids = as.integer(factor(d$Subject)),
  n_subjects = 36,
  times = d$Time,
  N = nrow(d),
  ppc_t = c(0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0)
)

# Fit the model in stan
fit <- sampling(model, model_data, chains = 12, warmup = 2000, iter = 4000)


# ---- Recreate Figure 3 ----


# Various diagnostic plots.
pop_obs <-
  fit %>%
  spread_draws(pop_obs[i]) %>%
  mutate(pop_obs = 1000 * pop_obs) %>%
  mean_qi(.width = c(0.5, 0.8, 0.95)) %>%
  mutate(.width = scales::percent(.width))

ppc_plot <-
  pop_obs %>%
  left_join(d) %>%
  ggplot() +
  geom_interval(aes(
    Time,
    ymin = .lower,
    ymax = .upper,
    color = factor(.width)
  )) +
  geom_jitter(
    data = d,
    aes(Time, Concentration),
    height = 0,
    width = 0.2,
    alpha = 0.25
  ) +
  scale_color_brewer(palette = 'OrRd', direction = -1) +
  labs(x = 'Time Post Dose', y = 'Concentration (ng/ml)', color = 'Posterior Probability') +
  theme(legend.position = c(0.75, 0.8)) +
  theme(aspect.ratio = 1) +
  guides(color = guide_legend(reverse = TRUE))


resids <-
  fit %>%
  spread_draws(C[i]) %>%
  mutate(log_C = log(1000 * C)) %>%
  mean_qi() %>%
  left_join(d) %>%
  mutate(err = log(Concentration) - log_C)


err_plot <-
  resids %>%
  ggplot(aes(log_C, err)) +
  geom_point(size = 2, fill = 'gray', shape = 21) +
  geom_hline(aes(yintercept = 0)) +
  geom_smooth(color = 'red', se = F, size = 1) +
  labs(x = 'Predicted Log Concentration', y = 'Residual on Log Scale') +
  theme(aspect.ratio = 1)

pred_plot <-
  fit %>%
  spread_draws(C[i]) %>%
  mutate(C = 1000 * C) %>%
  mean_qi() %>%
  left_join(d) %>%
  ggplot(aes(Concentration, C, ymin = .lower, ymax = .upper)) +
  geom_abline(color = 'black') +
  geom_pointrange(
    alpha = 0.5,
    color = 'black',
    fill = 'gray',
    shape = 21,
    size = 0.25
  ) +
  theme(aspect.ratio = 1) +
  labs(x = 'True Concentration (ng/ml)', y = 'Predicted Concentration (ng/ml)')


ecdf_plot <-
  fit %>%
  spread_draws(ppc_C[i], n = 100) %>%
  ungroup %>%
  mutate(C = 1000 * ppc_C) %>%
  select(C, .draw) %>%
  ggplot(aes(C, group = .draw)) +
  stat_ecdf(aes(color = 'Posterior Concentrations'), size = 1) +
  stat_ecdf(
    data = d,
    aes(Concentration, color = 'Observed Concentration'),
    inherit.aes = F,
    size = 1
  ) +
  scale_color_manual(values = c('black', 'gray')) +
  theme(aspect.ratio = 1,
        legend.position = c(0.75, 0.75)) +
  labs(x = 'Concentration (ng/ml)', y = 'Cumulative Probability', color = '')

figure_3 <- ((ppc_plot | pred_plot) / (err_plot | ecdf_plot))

ggsave(filename = 'figure_3.pdf', plot = figure_3, path = here("figures"))


# ---- Recreate figure 4 ----


# Need to identify which subjects have best/worst fit according to MAPE
# This will compute the error for each subject over the training data
error <- fit %>%
  spread_draws(C[i], n = 2000) %>%
  mean_qi() %>%
  left_join(d) %>%
  ungroup %>%
  select(Subject, C, Concentration_scaled) %>%
  group_by(Subject) %>%
  summarise(mape = mape(1000 * Concentration_scaled, 1000 * C)) %>%
  arrange(desc(mape))

# Get the best and worst fit
# Since error is arranged in descending order just take the first (worst) and last (bests) rows
candidate <- rbind(head(error, 1), tail(error, 1))
candidate$Rank <- c('Worst Fit', 'Best Fit')

best_worst_data <- d %>%
  left_join(candidate) %>%
  filter(Subject %in% candidate$Subject)

# Now plot the best and worst fits
data_plot <-  best_worst_data %>%
  ggplot(aes(Time, Concentration)) +
  geom_point(aes(fill = Rank), shape = 21, size = 2) +
  scale_fill_brewer(palette = 'Set1', direction = -1) +
  guides(fill = F) +
  labs(x = 'Time Post Dose',
       y = 'Concentration (ng/ml)',
       subtitle = 'Observed Concentration') +
  theme(aspect.ratio = 1) +
  ylim(0, 250)

# Finally, plot predictions for the best and worst fits.
posterior_plot_data <- fit %>%
  spread_draws(C[i], n = 125, seed = 0) %>%
  inner_join(d) %>%
  inner_join(candidate, by = 'Subject') 
  
  
posterior_plot <- posterior_plot_data %>% 
                  ggplot(aes(Time, 1000 * C, color = Rank, group = interaction(.draw, Rank))) +
                  geom_line(alpha = 0.2) +
                  geom_point(
                    data = best_worst_data,
                    aes(Time, Concentration, fill = Rank),
                    shape = 21,
                    size = 2,
                    inherit.aes = F) +
                  scale_fill_brewer(palette = 'Set1', direction = -1) +
                  scale_color_brewer(palette = 'Set1', direction = -1) +
                  theme(aspect.ratio = 1,
                        legend.position = c(0.8, 0.8)) +
                  ylim(0, 250) +
                  labs(
                    x = 'Time Post Dose',
                    y = 'Concentration (ng/ml)',
                    subtitle = 'Plausible Concentrations\nAfter Seeing Data',
                    color = '',
                    fill = '')


figure_4 <- prior_plot + data_plot + posterior_plot

ggsave(filename = 'figure_4.pdf', plot = figure_4, path = here("figures"))


# ---- Posterior predictive by subject ----
# This figure is not appear in the paper but is worth generating anyway
# This is the posterior predictive distribution for each subject
# We plot the posterior predictive intervals for the latent concentration and the observed concentration
by_subject_ppc_data <-
  fit %>%
  spread_draws(C[i], ppc_C[i]) %>%
  mean_qi() %>%
  left_join(d)

by_subject_ppc <-
  by_subject_ppc_data %>%
  ggplot() +
  geom_line(aes(Time, C, color = 'Predictions')) +
  geom_ribbon(aes(
    Time,
    ymin = ppc_C.lower,
    ymax = ppc_C.upper,
    alpha = 'PPC Concentration'
  ),fill = 'red') +
  geom_ribbon(aes(
    Time,
    ymin = C.lower,
    ymax = C.upper,
    alpha = 'Latent Concentration'
  ),fill = 'red') +
  geom_point(aes(Time, Concentration_scaled, color = 'Observed Data'), size = 0.5) +
  facet_wrap( ~ Subject, scales = 'free_y') +
  scale_alpha_manual(values = c(0.5, 0.25)) +
  scale_color_manual(values = c('black', 'red')) +
  labs(alpha = '', color = '')+
  theme(aspect.ratio = 1/1.61,
        legend.position = 'top')

ggsave(filename = 'by_subject_ppc.pdf',
       plot = by_subject_ppc,
       path = here("figures"),
       height = 8, width = 15)



# ---- Generate Pseudodata ----

#Generate the pseudodata to be used in the next step.
# Generate 250 subjects, we can use less if we want.
N_subjects <- 250

#Times at which to observe the subjects.
#Will splice intro train and test later
sample_times <- seq(0.5, 12, 0.25)
ids <- 1:N_subjects

#Easy way to get all times for all patients.
pairs <- crossing(ids, sample_times)

#Prepare model data
subjectids <- pairs$ids
times <- pairs$sample_times
N <- length(times)

model_data <- list( N = N, times = times, subjectids = subjectids, N_subjects = N_subjects)

model <- here('models', 'generate_pseudo_data.stan') %>% 
         stan_model()

# Actually perform draws
fits <- sampling(model, data = model_data, iter = 1, chains = 1, algorithm = 'Fixed_param', seed = 19920908)

samples <- rstan::extract(fits)

saveRDS(samples, here('data', 'simulated_data.RDS'))


# Will need the true pk parameters at the final step to compute calibration.
# Save in a csv.
data <-
  tibble(
  subjectids = subjectids,
  times = times,
  C = as.numeric(samples$C),
  Cobs = as.numeric(samples$Cobs)
  ) %>%
  left_join(
    tibble(
      subjectids = unique(subjectids),
      tmax = as.numeric(samples$tmax),
      cl = as.numeric(samples$Cl),
      ke = as.numeric(samples$ke),
      ka = as.numeric(samples$ka),
      alpha = as.numeric(samples$alpha))
  )

write_csv(data, here('data', 'simulated_data.csv'))
