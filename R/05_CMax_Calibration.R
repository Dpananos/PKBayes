library(tidyverse)
library(rray)
library(tictoc)
library(patchwork)
library(here)
theme_set(theme_minimal(base_size = 12))

map_draw_location = here('data', 'map_parameter_draws.RDS')
map_draws = readRDS(map_draw_location)

mcmc_draw_location =  here('data', 'mcmc_parameter_draws.RDS')
mcmc_draws = readRDS(mcmc_draw_location)

#Dose sizes
DD = c(seq(0,10, 0.05),seq(11, 60, 0.5))
nD = length(DD)

# Function to evaluate risk of being below threshold at tmax
# This function uses the parameter draws from the posterior to compute the concentration curve
# Then, we can evaluate curves over dose sizes.
# Returns an array of risk which corresponds to dose size
get_risk_at_max = function(patient, draws, thresh){
  n = nrow(draws[[1]])
  
  #Pk paramters for patient
  ka = rray(draws$ka[,patient], dim=c(n,1))
  ke = rray(draws$ke[,patient], dim=c(n,1))
  cl = rray(draws$cl[,patient], dim=c(n,1))
  time = log(ka/ke)/(ka - ke)
  
  # Dose sizes to evaluate over
  D = rray(DD, dim = c(1,nD))
  
  # Array broadcasting for economy of thought
  y = 1000*(ke*ka)/(2*cl*(ke-ka))*(exp(-ka*time) - exp(-ke*time))
  yy = y*D
  r= yy %>% rray_lesser(thresh) %>% rray_mean(axes = c(1))
  
  as.numeric(r)
}

estimate_d = function(model, p){
  roots = uniroot(function(x) model(x) - p, c(0, max(DD)))
  roots$root
}


risk_at_max<-
  tibble(patient = 1:100) %>% 
  mutate(
    #Get risk for each patient
    map_risk_at_max = map(patient, ~get_risk_at_max(.x, map_draws, 100)),
    mcmc_risk_at_max = map(patient, ~get_risk_at_max(.x, mcmc_draws, 100)),
    # Interpolate the risk as a function of dose using a hermite spline
    D = list(DD),
    mcmc_spline = map2(D, mcmc_risk_at_max, ~splinefun(.x, .y, method='hyman')),
    map_spline = map2(D, map_risk_at_max, ~splinefun(.x, .y, method='hyman'))
  )



doses_for_max = risk_at_max %>% 
  select(patient, mcmc_spline, map_spline) %>% 
  crossing(p = seq(0.05, 0.95, 0.05)) %>% 
  # This line uses root solving to find the dose required to achieve the desired risk level
  mutate(mcmc_estimated_dose =  map2_dbl(mcmc_spline, p, estimate_d),
         map_estimated_dose =  map2_dbl(map_spline, p, estimate_d),
         delta = map_estimated_dose - mcmc_estimated_dose
  )

doses_for_max%>% 
  select(patient, p, mcmc_estimated_dose, map_estimated_dose) %>% 
  write_csv(here("data","experiment_2_doses.csv"))


figure_7_right = doses_for_max %>% 
  ggplot(aes(p, delta, group = patient))+
  geom_line()+
  scale_x_continuous(labels = scales::percent, limits = c(0,0.5))+
  geom_hline(aes(yintercept = 0), color = 'red')

ggsave(filename = 'figure_7_right.pdf',
       plot = figure_7_right,
       path = here('figures'))


# ---- Calibration ----


dose_location = here( "data","experiment_2_doses.csv")
doses_for_max %>% 
  select(patient, p, mcmc_estimated_dose, map_estimated_dose) %>% 
  write_csv(dose_location)



experiment_2_doses = read_csv(here("data","experiment_2_doses.csv")) %>% 
  filter(p<=0.5)


pkfunc<-function(dose, cl, ke, ka, t){
  
  1000*dose*ke*ka/(2*cl*(ke - ka))*(exp(-ka*t) - exp(-ke*t))
  
}



# To determine calibration, we give each patient their recommended dose for the desired risk
# Each dose was designed to elicit a risk of exceeding some threshold
# The calibration is the propotion of those patients who fail to exceed the threshold.
figure_8_right_data = experiment_2_doses %>% 
  left_join(true_pk_params, by = c('patient' = 'subjectids')) %>% 
  mutate(t = log(ka/ke)/(ka - ke), 
         conc_mcmc = pkfunc(mcmc_estimated_dose, cl, ke, ka, t),
         conc_map = pkfunc(map_estimated_dose, cl, ke, ka, t),
         calibration_mcmc = conc_mcmc<=100,
         calibration_map = conc_map<=100) %>%
  group_by(p) %>% 
  summarize(mcmc_calib = mean(calibration_mcmc), 
            map_calib = mean(calibration_map))


figure_8_right<-figure_8_right_daya
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


ggsave(filename = 'figure_8_right.pdf',
       plot = figure_8_right,
       path = here("figures"))

