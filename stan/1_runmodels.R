library(rstan)
library(bayesplot)
library(tidyverse)
library(tidybayes)
library(ggpubr)
library(Metrics)
library(patchwork)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)
theme_set(theme_minimal(base_size = 12))




# ---- Prior Predictive ----

d = read_csv('data/apixiban_regression_data.csv') %>% 
    mutate(i = seq_along(Time))


prior_model = stan_model('stan/models/prior_predictive_check.stan')

ts = seq(0.5, 12, 0.1)
prior_params = list(
  N = length(ts),
  t = ts,

  
  MU_CL_MEAN = log(log(3.3)),
  MU_CL_SIGMA = 0.25,
  
  MU_T_MEAN = log(3.3),
  MU_T_SIGMA = 0.25,
  
  S_T_A = 10,
  S_T_B = 100,

  S_CL_A = 15,
  S_CL_B = 100,
  
  PHI_A = 20,
  PHI_B = 20,
  KAPPA_A = 20,
  KAPPA_B = 20,
  
  SIGMA_MEAN = log(0.2),
  SIGMA_SIGMA = 0.2
)

prior_samples = sampling(prior_model, 
                         data = prior_params, 
                         algorithm='Fixed_param',
                         chains=1,
                         iter=2000)



prior = rstan::extract(prior_samples)


tframe = tibble(t=ts, i = seq_along(ts))

prior_samples %>% 
  spread_draws(y[i]) %>% 
  left_join(tframe) %>% 
  ggplot()+
  geom_line(aes(t,1000*y, group=.draw), alpha = 0.05)+
  theme_classic()+
  labs(x='Time (Hours after ingestion)', y='Concentration (mg/L)')

p<-prior_samples %>% 
  spread_draws(Observations[i]) %>% 
  mutate(Observations = 1000*Observations) %>% 
  left_join(tframe) %>% 
  group_by(t) %>% 
  mean_qi(Observations, .width = c(0.5, 0.8, 0.95)) %>% 
  ungroup %>% 
  mutate(.width = scales::percent(.width)) %>% 
  ggplot(aes(t,Observations))+
  geom_interval()+
  scale_color_brewer(palette ='OrRd')+
  theme_classic()+
  labs(x='Hours Post Dose', 
       y='Prior Predictions of\nObserved Concentrations (ng/ml)',
       color='Credible Intervals')+
  theme(aspect.ratio = 1/1.61)

  
#### -----MODELLING----- ####


model = stan_model('stan/models/model.stan')

model_data = list(
  yobs = d$Concentration_scaled,
  subjectids = as.integer(factor(d$Subject)),
  n_subjects = 36,
  times = d$Time,
  N = nrow(d),
  ppc_t = c(0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0)
)


fit = sampling(model,
               model_data, 
               chains = 12, 
               warmup = 2000, 
               iter = 4000)

p = rstan::extract(fit)

ppc = fit %>% 
  spread_draws(pop_obs[i]) 


#### Bayes in 3 panels ####

prior_plot<-prior_samples %>% 
  spread_draws(y[i], n = 250, seed = 1) %>% 
  left_join(tframe) %>% 
  ggplot(aes(t, 1000*y, group = .draw))+
  geom_line(alpha = 0.2)+
  labs(x = 'Hours Post Dose',
       y = 'Concentration (ng/ml)',
       subtitle = 'Plausible Concentrations\nBefore Seeing Data')+
  ylim(0,250)+
  theme(aspect.ratio = 1)


error = fit %>% 
  spread_draws(C[i], n = 2000) %>% 
  mean_qi() %>% 
  left_join(d) %>% 
  ungroup %>% 
  select(Subject, C, Concentration_scaled) %>% 
  group_by(Subject) %>% 
  summarise(mape = mape(1000*Concentration_scaled, 1000*C)) %>% 
  arrange(desc(mape))

candidate = rbind(head(error,1), tail(error,1))
candidate$Rank = c('Worst Fit', 'Best Fit')
best_worst_data = d %>% 
  left_join(candidate) %>% 
  filter(Subject %in% candidate$Subject)

data_plot =  best_worst_data %>% 
  ggplot(aes(Time, Concentration))+
  geom_point(aes(fill = Rank), shape = 21, size = 2)+
  scale_fill_brewer(palette = 'Set1', direction = -1)+
  guides(fill = F)+
  labs(x = 'Time Post Dose',
       y = 'Concentration (ng/ml)',
       subtitle = 'Observed Concentration')+
  theme(aspect.ratio = 1)+
  ylim(0,250)


posterior_plot = fit %>% 
  spread_draws(C[i], n = 125, seed = 0) %>% 
  inner_join(d) %>% 
  inner_join(candidate, by = 'Subject') %>% 
  ggplot(aes(Time, 1000*C, color = Rank, group = interaction(.draw, Rank)))+
  geom_line(alpha = 0.2)+
  geom_point(data = best_worst_data,
             aes(Time, Concentration, fill = Rank),
             shape = 21,
             size = 2,
             inherit.aes = F)+
  scale_fill_brewer(palette = 'Set1', direction = -1)+
  scale_color_brewer(palette = 'Set1', direction = -1)+
  theme(aspect.ratio = 1,
        legend.position = c(0.8, 0.8))+
  ylim(0,250)+
  labs(x = 'Time Post Dose',
       y = 'Concentration (ng/ml)',
       subtitle = 'Plausible Concentrations\nAfter Seeing Data',
       color = '',
       fill = '')


fig = (prior_plot + data_plot + posterior_plot)
ggsave(filename = 'figs/bayes_in_3_model.png',plot = fig,  height = 3, width = 9)

#### Diagnostics ####

pop_obs = fit %>% 
  spread_draws(pop_obs[i]) %>% 
  mutate(pop_obs = 1000*pop_obs) %>% 
  mean_qi(.width = c(0.5, 0.8, 0.95)) %>% 
  mutate(.width = scales::percent(.width))

ppc_plot = pop_obs %>%
  left_join(d) %>% 
  ggplot()+
  geom_interval(aes(Time, ymin = .lower, ymax = .upper, color = factor(.width)))+
  geom_jitter(data = d, aes(Time, Concentration), height = 0, width = 0.2, alpha = 0.25)+
  scale_color_brewer(palette = 'OrRd', direction = -1)+
  labs(x = 'Time Post Dose', y = 'Concentration (ng/ml)', color = 'Posterior Probability')+
  theme(legend.position = c(0.75, 0.8))+
  theme(aspect.ratio = 1)+
  guides(color = guide_legend(reverse = TRUE))


resids = fit %>% 
  spread_draws(C[i]) %>% 
  mutate(log_C = log(1000*C)) %>% 
  mean_qi() %>% 
  left_join(d) %>% 
  mutate(err = log(Concentration) - log_C)


err_plot = resids %>% 
  ggplot(aes(log_C, err))+
  geom_point(size = 2, fill = 'gray', shape = 21)+
  geom_hline(aes(yintercept = 0))+
  geom_smooth(color = 'red', se = F, size = 1)+
  labs(x = 'Predicted Log Concentration', y = 'Residual on Log Scale')+
  theme(aspect.ratio = 1)

pred_plot = fit %>% 
  spread_draws(C[i]) %>% 
  mutate(C = 1000*C) %>% 
  mean_qi() %>% 
  left_join(d) %>% 
  ggplot(aes(Concentration, C, ymin = .lower, ymax = .upper))+
  geom_abline(color = 'black')+
  geom_pointrange(alpha = 0.5, color = 'black', fill = 'gray', shape = 21, size = 0.25)+
  theme(aspect.ratio = 1)+
  labs(x = 'True Concentration (ng/ml)', y = 'Predicted Concentration (ng/ml)')

  
ecdf_plot = fit %>% 
  spread_draws(ppc_C[i], n = 100) %>% 
  ungroup %>% 
  mutate(C = 1000*ppc_C) %>% 
  select(C, .draw) %>% 
  ggplot(aes(C, group = .draw))+
  stat_ecdf(aes(color = 'Posterior Concentrations'), size = 1)+
  stat_ecdf(data = d, aes(Concentration, color = 'Observed Concentration'), inherit.aes = F, size = 1)+
  scale_color_manual(values = c('black','gray'))+
  theme(aspect.ratio = 1,
        legend.position = c(0.75, 0.75))+
  labs(x = 'Concentration (ng/ml)', y = 'Cumulative Probability', color = '')
  



fig2 = ((ppc_plot | pred_plot) / (err_plot | ecdf_plot))

ggsave('figs/diagnostics.png', fig2, width = 8, height = 8)
