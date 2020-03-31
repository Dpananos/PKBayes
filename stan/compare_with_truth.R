library(tidyverse)
library(ggstance)
library(GGally)

hmc = read_csv('data/hmc_patient_parameter_estimates.csv') %>% mutate(kind = 'HMC') 
mp = read_csv('data/map_patient_parameter_estimates.csv') %>% mutate(kind = 'MAP')

  
  
true = read_csv('data/simulated_data.csv') %>% 
  rename(subject = subjectids) %>% 
  distinct(subject, .keep_all = T) %>% 
  select(subject, tmax:alpha) %>% 
  rename(t = tmax) %>% 
  gather(.variable, .value, -subject) %>% 
  mutate(kind = 'Truth')  %>% 
  mutate(.variable = if_else(.variable=='cl', 'Cl',.variable)) %>% 
  filter(subject<=100)


f =  c(1,9,12,18,39,41,46,49,57,63,69,75,76,83,88,91,93,94,99)

hmc %>% 
  bind_rows(mp) %>% 
  bind_rows(true)  %>% 
  filter(subject %in% f, .variable=='alpha') %>% 
  select(subject, .lower, kind) %>% 
  spread(kind,.lower)
  # 
  # mutate(.variable = stringr::str_to_lower(.variable)) %>% 
  # ggplot(aes(.value, factor(subject), xmin = .lower, xmax = .upper, color = kind))+
  # geom_pointrange(position = position_dodgev(height = 0.5), size = 0.25)+
  # facet_wrap(~.variable, scales = 'free_x')
  # 
  # 
  # 
