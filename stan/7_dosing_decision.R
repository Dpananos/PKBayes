library(tidyverse)
library(rray)
library(tictoc)
library(patchwork)
theme_set(theme_minimal(base_size = 12))

map_draws = readRDS('data/map_parameter_draws.RDS') 
mcmc_draws = readRDS('data/mcmc_parameter_draws.RDS')
DD = c(seq(0,10, 0.05),seq(11, 60, 0.5))
nD = length(DD)

get_risk = function(i, draws, thresh){
  n = nrow(draws[[1]])
  ka = rray(draws$ka[,i], dim=c(n,1))
  ke = rray(draws$ke[,i], dim=c(n,1))
  cl = rray(draws$cl[,i], dim=c(n,1))
  time = rray(seq(0.75,11.75,length.out = 23), dim = c(1,23))
  
  D = rray(DD, dim = c(nD,1))
  
  y = 1000*(ke*ka)/(2*cl*(ke-ka))*(exp(-ka*time) - exp(-ke*time))
  yy = D*rray_reshape(y, dim = c(1, n, 23))
  r= yy %>% rray_lesser(thresh) %>% rray_mean(axes = c(2,3))
  
  as.numeric(r)
}

get_risk_at_12 = function(i, draws, thresh){
  n = nrow(draws[[1]])
  ka = rray(draws$ka[,i], dim=c(n,1))
  ke = rray(draws$ke[,i], dim=c(n,1))
  cl = rray(draws$cl[,i], dim=c(n,1))
  time = 12
  
  D = rray(DD, dim = c(1,nD))
  
  y = 1000*(ke*ka)/(2*cl*(ke-ka))*(exp(-ka*time) - exp(-ke*time))
  yy = y*D
  r= yy %>% rray_lesser(thresh) %>% rray_mean(axes = c(1))
  
  as.numeric(r)
}


estimate_d = function(model, p){
  roots = uniroot(function(x) model(x) - p, c(0, max(DD)))
  roots$root
}
 


# Analysis
tic()
risk = tibble(i = 1:100) %>% 
  mutate(
    map_risk = map(i, ~get_risk(.x, map_draws, 20)),
    mcmc_risk = map(i, ~get_risk(.x, mcmc_draws,  20)),
    map_risk_12 = map(i, ~get_risk_at_12(.x, map_draws, 20)),
    mcmc_risk_12 = map(i, ~get_risk_at_12(.x, mcmc_draws, 20))
      
      )
toc()



risk = risk %>% 
  mutate(
    D = list(DD),
    mcmc_model = map2(D, mcmc_risk_12, ~splinefun(.x, .y, method='hyman')),
    map_model = map2(D, map_risk_12, ~splinefun(.x, .y, method='hyman')),
    mcmc_D = map2_dbl(mcmc_model, 0.50, estimate_d),
    map_D = map2_dbl(map_model, 0.50, estimate_d)
  )



models = risk %>% select(i, mcmc_model, map_model)

models %>% 
  crossing(p = seq(0.05,0.95,0.05)) %>% 
  mutate(mcmc_d = map2_dbl(mcmc_model, p, estimate_d),
         map_d = map2_dbl(map_model, p, estimate_d),
         deltad = map_d - mcmc_d) %>% 
  filter(p <=0.5) %>% 
  select(i, p, mcmc_d, map_d) %>% 
  write_csv('data/experiment_1_doses.csv')


#Plot for C at 12 hour experiment
plot1 = models %>% 
  crossing(p = seq(0.05,0.95,0.05)) %>% 
  mutate(mcmc_d = map2_dbl(mcmc_model, p, estimate_d),
         map_d = map2_dbl(map_model, p, estimate_d),
         deltad = map_d - mcmc_d) %>% 
  arrange(deltad) %>% 
  ggplot(aes(p, deltad, group=i))+
  geom_line(alpha = 0.5)+
  geom_hline(aes(yintercept = 0), color = 'red')+
  scale_x_continuous(labels = scales::percent, breaks = seq(0, 0.5, 0.1), limits = c(0.05,0.5))+
  scale_y_continuous(breaks = seq(-3,3), limits = c(-3,3))+
  ylab('MAP Dose - HMC Dose')+
  xlab('Risk At 12 Hours Post Dose')+
  theme(aspect.ratio = 1/1.61)+
  ggsave('figs/risk_dose_1.png',height = 3, width = 5)


#plot dose sizes
# dev.new()
# options(device = "quartz")
# models %>% 
#   crossing(p = seq(0.05,0.95,0.05)) %>% 
#   mutate(mcmc_d = map2_dbl(mcmc_model, p, estimate_d),
#          map_d = map2_dbl(map_model, p, estimate_d),
#          deltad = map_d - mcmc_d) %>% 
#   arrange(deltad) %>% 
#   select(i, map_d, mcmc_d, p) %>% 
#   gather(method, dose, -i, -p) %>% 
#   ggplot(aes(p, dose, color = method, group = interaction(method, i)))+
#   geom_line()+
#   facet_wrap(~i, scales = 'free_y')
#   
  

#Plot for whole 12 hour experiment

risk = risk %>% 
  mutate(
    D = list(DD),
    mcmc_model = map2(D, mcmc_risk, ~splinefun(.x, .y, method='hyman')),
    map_model = map2(D, map_risk, ~splinefun(.x, .y, method='hyman')),
    mcmc_D = map2_dbl(mcmc_model, 0.50, estimate_d),
    map_D = map2_dbl(map_model, 0.50, estimate_d)
  )

models = risk %>% select(i, mcmc_model, map_model)

models %>% 
  crossing(p = seq(0.05,0.95,0.05)) %>% 
  mutate(mcmc_d = map2_dbl(mcmc_model, p, estimate_d),
         map_d = map2_dbl(map_model, p, estimate_d),
         deltad = map_d - mcmc_d) %>% 
  filter(p <=0.5) %>% 
  select(i,p, mcmc_d, map_d) %>% 
  write_csv('data/experiment_2_doses.csv')

plot2 = models %>% 
  crossing(p = seq(0.05,0.95,0.05)) %>% 
  mutate(mcmc_d = map2_dbl(mcmc_model, p, estimate_d),
         map_d = map2_dbl(map_model, p, estimate_d),
         deltad = map_d - mcmc_d) %>% 
  ggplot(aes(p, deltad, group=i))+
  geom_line(alpha = 0.5)+
  geom_hline(aes(yintercept = 0), color = 'red')+
  scale_x_continuous(labels = scales::percent, breaks = seq(0.0, 0.5, 0.1), limits = c(0.05, 0.5))+
  scale_y_continuous(breaks = seq(-3,3), limits = c(-3,3))+
  ylab('MAP Dose - HMC Dose')+
  xlab('Risk Over 12 Hour Observation Period')+
  theme(aspect.ratio = 1/1.61)+
  ggsave('figs/risk_dose_2.png',height = 3, width = 5)



fig = (plot1+plot2)  

ggsave('mlhc-submission-files/figs/experiments.png', 
       fig,
       height = 3,
       width = 7)




