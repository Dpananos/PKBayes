library(rstan)
library(bayesplot)
library(tidyverse)
library(tidybayes)
source('stan/stan_utilities.R')
options(mc.cores = parallel::detectCores())


#### GENERATE PRIORS ####

d = read_csv('~/Documents/PhD Code/PKBayes/data/apixiban_regression_data.csv')


prior_model = stan_model('stan/models/prior_predictive_check.stan')

prior_params = list(
  N = 8,
  t = c( 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0),

  
  MU_CL_MEAN = log(log(3.3)),
  MU_CL_SIGMA = 0.3,
  
  MU_T_MEAN = log(3.3),
  MU_T_SIGMA = 0.5,
  
  S_T_A = 10,
  S_T_B = 100,

  S_CL_A = 15,
  S_CL_B = 100,
  
  PHI_A = 20,
  PHI_B = 20,
  KAPPA_A = 20,
  KAPPA_B = 20
)

prior_samples = sampling(prior_model, 
                         data = prior_params, 
                         algorithm='Fixed_param',
                         chains=1,
                         iter=2000)



prior = rstan::extract(prior_samples)


tframe = tibble(t=c( 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0), i = seq_along(t))

prior_samples %>% 
  spread_draws(y[i]) %>% 
  left_join(tframe) %>% 
  ggplot()+
  geom_line(aes(t,1000*y, group=.draw), alpha = 0.05)+
  theme_classic()+
  labs(x='Time (Hours after ingestion)', y='Concentration (ng/mL)')

p<-prior_samples %>% 
  spread_draws(y[i]) %>% 
  left_join(tframe) %>% 
  group_by(t) %>% 
  median_hdci(y, .width = seq(0.5, 0.95,0.05)) %>%  
  ungroup %>% 
  mutate(.alpha = 1-.width) %>% 
  ggplot(aes(t,y))+
  geom_ribbon(aes(ymin=.lower, ymax=.upper, group=.width, alpha= .alpha))+
  geom_line(aes(color = 'Prior Mean'))+
  scale_color_manual(values = c('red'))+
  scale_alpha_identity()+
  guides(alpha=F)

p +
  geom_point(data = d, aes(Time,Concentration_scaled), color ='orange')+
  theme_classic()
  
#### -----MODELLING----- ####

model = stan_model('stan/models/model.stan')

d = read_csv('~/Documents/PhD Code/PKBayes/data/apixiban_regression_data.csv')

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


preds = fit %>% 
  spread_draws(C[i], C_ppc[i]) %>% 
  left_join(d %>% select(Time,Subject) %>% mutate(i=1:n())) %>% 
  group_by(Time, Subject) %>% 
  mean_hdi(C, C_ppc) 


preds %>% 
  ungroup %>% 
  left_join(d) %>% 
  ggplot(aes(Time, C))+
  geom_ribbon(aes(ymin=C.lower, ymax=C.upper),alpha=0.5)+
  geom_ribbon(aes(ymin=C_ppc.lower, ymax=C_ppc.upper),alpha=0.25)+
  geom_line()+
  geom_point(aes(Time, Concentration_scaled), inherit.aes = F, size = 1, color= 'orange')+
  facet_wrap(~Subject, scales = 'free_y')



bayesplot::mcmc_areas(fit, regex_pars = 'ppc_tmax',prob = 0.95)
