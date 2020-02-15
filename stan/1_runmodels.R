library(rstan)
library(bayesplot)
library(tidyverse)
library(tidybayes)
library(ggpubr)
options(mc.cores = parallel::detectCores())

theme_set(theme_classic())




#### GENERATE PRIORS ####

d = read_csv('data/apixiban_regression_data.csv')


prior_model = stan_model('stan/models/prior_predictive_check.stan')

ts = c(0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0)
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

p
ggsave('figs/prior_predictions.png', height = 4)

p +
  geom_point(data = d, aes(Time, Concentration), color = 'black', size = 0.25)
  
#### -----MODELLING----- ####

d = read_csv('data/apixiban_regression_data.csv')

model = stan_model('stan/models/model.stan')

model_data = list(
  yobs = d$Concentration_scaled,
  subjectids = as.integer(factor(d$Subject)),
  n_subjects = 36,
  times = d$Time,
  N = nrow(d)
)


fit = sampling(model,
               model_data, 
               chains = 12, 
               warmup = 2000, 
               iter = 4000)

p = rstan::extract(fit)


####--- Plots ----

ppc = fit %>% 
  spread_draws(pop_obs[i]) 


# Population level observations
pop_obs_plot = ppc %>% 
  mutate(pop_obs = pop_obs*1000) %>% 
  mean_qi(.width=c(0.5, 0.8, 0.95)) %>% 
  left_join(tframe) %>% 
  ggplot(aes(t, pop_obs))+
  geom_interval()+
  geom_jitter(data = d, 
              aes(Time, Concentration),
              size = 1, alpha = 0.25, width = 0.1, height = 0) +
  scale_color_brewer()+
  theme_classic()+
  labs(x='Hours Post Dose', 
       y='Posterior Population Predictions of\nObserved Concentrations',
       color='Credible Intervals')+
  theme(aspect.ratio = 1)

# Observed vs Predicted

preds = fit %>% 
  spread_draws(C[i], ppc_C[i], n=2000) %>% 
  mean_qi() %>% 
  bind_cols(d) 

err_plot = preds %>% 
  mutate(C = 1000*C) %>% 
  ggplot(aes(C, Concentration - C)) +
  geom_point(color = 'black', fill = 'gray',pch=21)+
  geom_hline(aes(yintercept=0))+
  labs(x='Predicted Concentration (ng/ml)', y='Residual (ng/ml)')+
  theme(aspect.ratio = 1)+
  geom_smooth(se = F, color = 'red', size =1)

figure = ggarrange(pop_obs_plot, 
          err_plot,
          ncol=2, common.legend = T
          )
  ggsave('figs/model_results.png',plot = figure, height = 4, width = 7)

  

preds %>%
  ggplot(aes(Time,C))+
  geom_line(color='red')+
  geom_point(aes(Time, Concentration_scaled), inherit.aes = F, size = 1)+
  geom_ribbon(aes(ymin=C.lower, ymax=C.upper), alpha=0.5, fill='red')+
  geom_ribbon(aes(ymin=ppc_C.lower, ymax=ppc_C.upper), alpha=0.25, fill='red')+
  facet_wrap(~Subject, scales = 'free_y')+
  theme(aspect.ratio = 1/1.61)


fit %>% 
  gather_draws(ppc_tmax, mu_tmax, ppc_cl, mu_cl, ppc_ka, ppc_ke) %>% 
  mutate(.value = if_else(grepl('mu',.variable), exp(.value), .value)) %>% 
  ggplot(aes(.value))+
  geom_histogram(aes(y=..density..),color = 'black', fill = 'light gray')+
  facet_wrap(~.variable, scales = 'free')
