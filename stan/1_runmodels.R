library(rstan)
library(bayesplot)
library(tidyverse)
library(tidybayes)
options(mc.cores = parallel::detectCores())


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
  geom_line(aes(t,y, group=.draw), alpha = 0.05)+
  theme_classic()+
  labs(x='Time (Hours after ingestion)', y='Concentration (mg/L)')

p<-prior_samples %>% 
  spread_draws(Observations[i]) %>% 
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
       y='Prior Predictions of\nObserved Concentrations',
       color='Credible Intervals')+
  ylim(0.0, 0.25)


p +
  geom_point(data = d, aes(Time, Concentration_scaled), color = 'black', size = 0.25)
  
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


ppc = fit %>% 
  spread_draws(pop_obs[i]) 


ppc %>% 
  mean_qi(.width=c(0.5, 0.8, 0.95)) %>% 
  left_join(tframe) %>% 
  ggplot(aes(t, pop_obs))+
  geom_interval()+
  scale_color_brewer(palette ='OrRd')+
  theme_classic()+
  labs(x='Hours Post Dose', 
       y='Prior Predictions of\nObserved Concentrations',
       color='Credible Intervals')+
  theme(aspect.ratio = 1/1.61)


s = d %>% distinct(Subject, .keep_all = T) %>% mutate(i = 1:n())
fit %>% 
  spread_draws(z_CL[i]) %>% 
  mean_qi() %>% 
  inner_join(s) %>% 
  ggplot(aes(Weight, z_CL, ymin=.lower, ymax = .upper, color=Sex))+
  geom_pointrange()+
  theme(aspect.ratio = 1)



preds = fit %>% 
  spread_draws(C[i], ppc_C[i]) %>% 
  mean_qi()


preds %>% 
  bind_cols(d) %>% 
  ggplot(aes(Time,C))+
  geom_line(color='red')+
  geom_point(aes(Time, Concentration_scaled), inherit.aes = F, size = 1)+
  geom_ribbon(aes(ymin=C.lower, ymax=C.upper), alpha=0.5, fill='red')+
  geom_ribbon(aes(ymin=ppc_C.lower, ymax=ppc_C.upper), alpha=0.25, fill='red')+
  facet_wrap(~Subject, scales = 'free_y')+
  theme(aspect.ratio = 1/1.61)

  
fit %>% 
  spread_draws(ppc_cmax) %>% 
  ggplot(aes('cmax', ppc_cmax))+
  stat_interval()+
  scale_color_brewer(palette = 'OrRd')+
  theme_classic()


bayesplot::mcmc_hist(fit, regex_pars = 'mu', transformations = exp)
  
bayesplot::mcmc_hist(fit, regex_pars = 's_')
