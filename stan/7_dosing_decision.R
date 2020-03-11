library(tidyverse)
library(rray)
library(tictoc)
theme_set(theme_minimal(base_size = 15))

map_draws = readRDS('data/map_parameter_draws.RDS') 
mcmc_draws = readRDS('data/mcmc_parameter_draws.RDS')

get_risk = function(i, draws, thresh){
  n = nrow(draws[[1]])
  ka = rray(draws$ka[,i], dim=c(n,1))
  ke = rray(draws$ke[,i], dim=c(n,1))
  cl = rray(draws$cl[,i], dim=c(n,1))
  time = rray(seq(0.75,11.75,length.out = 23), dim = c(1,23))
  
  D = rray(seq(0,25, length.out = 51), dim = c(51,1))
  
  y = 1000*(ke*ka)/(2*cl*(ke-ka))*(exp(-ka*time) - exp(-ke*time))
  yy = D*rray_reshape(y, dim = c(1, n, 23))
  r= yy %>% rray_lesser(thresh) %>% rray_mean(axes = c(2,3))
  
  as.numeric(r)
}



estimate_d = function(model, p){
  roots = uniroot(function(x) model(x) - p, c(0, 25))
  roots$root
}



# Analysis
tic()
risk = tibble(i = 1:100) %>% 
  mutate(
    map_risk = map(i, ~get_risk(.x, map_draws, 20)),
    mcmc_risk = map(i, ~get_risk(.x, mcmc_draws,  20)))
toc()



risk = risk %>% 
  mutate(
    D = list(seq(0,25, length.out = 51)),
    mcmc_model = map2(D, mcmc_risk, ~splinefun(.x, .y, method='hyman')),
    map_model = map2(D, map_risk, ~splinefun(.x, .y, method='hyman')),
    mcmc_D = map2_dbl(mcmc_model, 0.75, estimate_d),
    map_D = map2_dbl(map_model, 0.75, estimate_d)
  )

risk %>% 
  ggplot(aes(map_D - mcmc_D))+
  geom_histogram(color = 'black', 
                 fill = 'light gray',
                 breaks = seq(-0.1, 0.3, 0.01))


models = risk %>% select(i, mcmc_model, map_model)

models %>% 
  crossing(p = seq(0.2,0.95,0.05)) %>% 
  mutate(mcmc_d = map2_dbl(mcmc_model, p, estimate_d),
         map_d = map2_dbl(map_model, p, estimate_d),
         deltad = map_d - mcmc_d) %>% 
  ggplot(aes(p, deltad, group=i))+
  geom_line(alpha = 0.5)+
  geom_hline(aes(yintercept = 0), color = 'red')+
  scale_x_reverse(labels = scales::percent, breaks = seq(0.2, 0.9, 0.2))+
  ylab('MAP Dose - MCMC Dose')+
  xlab('Proportion of time below threshold')+
  theme(aspect.ratio = 1/1.61)+
  ggsave('figs/risk_dose.png',height = 3, width = 5)
  
  

models %>% 
  mutate(x= list(seq(0,25, 0.01)),
         y = map2(map_model, x, ~.x(.y)) ) %>% 
  unnest(c(x,y)) %>% 
  ggplot(aes(x,y, group = i))+
  geom_line()

         