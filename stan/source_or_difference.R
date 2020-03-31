library(tidyverse)
library(tictoc)
library(patchwork)
library(tidybayes)
theme_set(theme_minimal(base_size = 12))

map_draws = readRDS('data/map_parameter_draws.RDS') 
mcmc_draws = readRDS('data/mcmc_parameter_draws.RDS')

get_risk = function(i, draws, thresh){
  
  ka = draws$ka[,i]
  ke = draws$ke[,i]
  cl = raws$cl[,i]
  
}


map_intervals = do.call(cbind, map_draws) %>% 
  gather_draws(ke[i], ka[i], Cl[i]) %>% 
  mean_qi %>% 
  mutate(kind = 'MAP')

mcmc_intervals = do.call( cbind, mcmc_draws) %>% 
  gather_draws(ke[i], ka[i], Cl[i]) %>% 
  mean_qi %>% 
  mutate(kind = 'MCMC')

map_intervals %>% 
  bind_rows(mcmc_intervals) %>% 
  filter(i %in% c(91)) %>% 
  ggplot(aes(factor(i), .value, ymin = .lower, ymax = .upper, color = kind))+
  geom_pointrange(position = position_dodge(width = 0.25))+
  facet_wrap(~.variable, nrow = 3, scales = 'free_y')




delta = map_intervals %>% 
  bind_rows(mcmc_intervals) %>% 
  select(.variable, i, .value, kind)  %>% 
  spread(kind, .value) %>% 
  ungroup %>% 
  transmute(delta = MCMC-MAP, .variable = .variable, i = i)


delta_width=map_intervals %>% 
  bind_rows(mcmc_intervals) %>% 
  mutate(width = .upper - .lower) %>% 
  select(.variable, i, width, kind) %>% 
  spread(kind, width) %>% 
  ungroup %>% 
  transmute(delta_width = MCMC-MAP, .variable = .variable, i = i)

f = delta_width %>% 
  inner_join(delta)

bad_ones =  c(1,9,12,18,39,41,46,49,57,63,69,75,76,83,88,91,93,94,99)

f %>% 
  ggplot(aes(delta, delta_width))+
  geom_point()+
  geom_point(data = f %>% filter(i %in% bad_ones ),
             aes(delta, delta_width), color = 'red')+
  facet_wrap(~.variable)+
  xlab('MCMC Point Estimate - MAP Point Estimate')+
  ylab('MCMC Interval Width - MAP Interval Width')



f %>% filter(i %in% bad_ones) %>% 
  arrange(delta_width)
