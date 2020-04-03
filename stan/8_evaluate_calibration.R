library(tidyverse)
library(patchwork)
theme_set(theme_minimal())

experiment_1_doses = read_csv('data/experiment_1_doses.csv') %>% 
  filter(p<=0.5)
experiment_2_doses = read_csv('data/experiment_2_doses.csv') %>% 
  filter(p<=0.5)
true_pk_params = read_csv('data/simulated_data.csv') %>% select(subjectids, cl, ke, ka) %>% 
  distinct(subjectids, .keep_all = T)


pkfunc<-function(dose, cl, ke, ka, t){
  
  1000*dose*ke*ka/(2*cl*(ke - ka))*(exp(-ka*t) - exp(-ke*t))
  
}



#Analyze decisions for experiment 1
cal1 = experiment_1_doses %>% 
  left_join(true_pk_params, by = c('i' = 'subjectids')) %>% 
  mutate(t = 12, 
         conc_mcmc = pkfunc(mcmc_d, cl, ke, ka, t),
         conc_map = pkfunc(map_d, cl, ke, ka, t),
         calibration_mcmc = conc_mcmc<=20,
         calibration_map = conc_map<=20) %>%
  group_by(p) %>% 
  summarize(mcmc_calib = mean(calibration_mcmc), 
            map_calib = mean(calibration_map)) %>% 
  ggplot()+
  geom_point(aes(p, mcmc_calib, color = 'HMC'))+
  geom_point(aes(p, map_calib, color = 'MAP'))+
  geom_abline()+
  theme(aspect.ratio = 1, legend.position = 'top')+
  scale_color_brewer(palette = 'Set1', direction = -1)+
  xlab('Desired Risk')+
  ylab('Calibration For Experiment 1')+
  scale_y_continuous(labels = scales::percent, limits = c(0,1))+
  scale_x_continuous(labels = scales::percent)+
  labs(color = '')
  
  

  

#Analyze decisions for experiment 2

cal2 = experiment_2_doses %>% 
  left_join(true_pk_params, by = c('i' = 'subjectids')) %>% 
  mutate(t =log(ka/ke)/(ka - ke), 
         conc_mcmc = pmap(list(dose = mcmc_d, cl = cl, ke = ke, ka = ka, t = t), pkfunc),
         conc_map = pmap(list(dose = map_d, cl = cl, ke = ke, ka = ka, t = t), pkfunc),
         calibration_mcmc = conc_mcmc<100,
         calibration_map = conc_map<100) %>% 
  group_by(p) %>% 
  summarize(mcmc_calib = mean(calibration_mcmc), 
            map_calib = mean(calibration_map)) %>% 
  ggplot()+
  geom_point(aes(p, mcmc_calib, color = 'HMC'))+
  geom_point(aes(p, map_calib, color = 'MAP'))+
  geom_abline()+
  theme(aspect.ratio = 1, legend.position = 'top')+
  scale_color_brewer(palette = 'Set1', direction = -1)+
  xlab('Desired Risk')+
  ylab('Calibration For Experiment 2')+
  scale_y_continuous(labels = scales::percent, limits = c(0,1))+
  scale_x_continuous(labels = scales::percent)+
  labs(color = '')

fig = cal1 + cal2

fig
ggsave('mlhc-submission-files/figs/fig8.png', fig,  height = 4.5, width = 1.61*4.5)
