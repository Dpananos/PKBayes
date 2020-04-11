suppressPackageStartupMessages(library(tidyverse))
library(ggpubr)
library(patchwork)
library(Metrics)
library(here)

theme_set(theme_minimal())
mcmc_predictions = here('data','mcmc_predictions.csv')
map_predictions = here('data','map_predictions.csv')
files = list(mcmc_predictions, map_predictions)

# This file contains out of sample predictions for MAP and HMC as well as true concentrations
# WE can use it to make plots as well as compute error metrics
d = map(files, read_csv) %>% 
  reduce(bind_rows) %>% 
  mutate(pred = 1000*pred, 
         low = 1000*low,
         high = 1000*high,
         C = 1000*C)


# ---- Make figure 5 ----

figure_5_left = d %>% 
  mutate(type = if_else(type == 'mcmc','HMC','MAP')) %>% 
  ggplot(aes(C, pred, color = type))+
  geom_point(alpha = 0.25)+
  geom_abline()+
  scale_x_log10()+
  scale_y_log10()+
  scale_color_brewer(palette = 'Set1', direction = -1)+
  theme(legend.position = 'top',
        aspect.ratio = 1)+
  labs(color = '',
       x = 'Latent log(Concentration)',
       y = 'Predicted Latent log(Concentration)')

figure_5_right = d %>% 
  select(times, subjectids, pred, type) %>% 
  mutate(type = if_else(type == 'mcmc','HMC','MAP')) %>% 
  spread(type, pred) %>% 
  ggplot(aes(HMC, MAP ))+
  geom_point(alpha = 0.25)+
  geom_abline(color = 'gray')+
  scale_x_log10()+
  scale_y_log10()+
  theme(aspect.ratio = 1)+
  labs(x='HMC Predicted\nLog Concentration (ng/ml)',
       y = 'MAP Predicted\nLog Concentration (ng/ml)')

figure_5 = figure_5_left + figure_5_right
ggsave(filename = 'figure_5.pdf',
       plot = figure_5,
       path = here('figures'))

subjects =  paste('Subject ', 1:100)

# Find pseudopatients which have a MAP credible interval
# 1.5 times as wide or wider than their HMC interval
bad_ones = d %>% 
  filter(between(times, 1, 5)) %>% 
  mutate(width = high - low) %>% 
  select(subjectids, times, width, type) %>% 
  spread(type, width) %>% 
  mutate(r = map/mcmc) %>% 
  group_by(subjectids) %>% 
  filter(r == max(r)) %>% 
  filter(r>=1.5) %>% 
  distinct(subjectids)

# ---- Error Metrics ----

losses<-d %>% 
  group_by(type) %>% 
  summarise(MSE = mse(C, pred),
            MAE = mae(C, pred),
            MAPE = mape(C, pred),
            MSE_SD = sd((C - pred)^2),
            MAE_SD = sd(abs(C - pred)),
            MAPE_SD = sd(abs(C-pred)/C)) %>% 
  mutate_if(is.numeric, function(x) round(x,2))


write_csv(losses, here('data','table_2_errors.csv'))

#---- Figure 6 ------


# Create figure 6
figure_6  = d %>% 
  mutate(type = stringr::str_to_upper(type),
         type = if_else(type == 'MCMC','HMC',type)) %>% 
  right_join(bad_ones) %>% 
  mutate(type = stringr::str_to_upper(type),
         subject = factor(paste('Subject ', subjectids), ordered = T, levels = subjects)
  ) %>% 
  ggplot(aes(times, pred, ymin=low, ymax=high, fill=forcats::fct_rev(type)))+
  geom_ribbon(alpha = 0.5)+
  facet_wrap(~subject, scales = 'free_y')+
  scale_fill_brewer(palette = 'Set1', direction = 1)+
  theme(legend.position = 'top', aspect.ratio = 1/1.61)+
  labs(x='Time (Hours Post Dose)', y='Concentration ng/mL', fill = '')


ggsave(filename = "figure_6.pdf",
       plot = figure_6,
       path = here('figures'))
       