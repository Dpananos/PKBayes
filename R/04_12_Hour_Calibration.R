suppressPackageStartupMessages(library(tidyverse))
library(rray)
library(patchwork)
library(here)
theme_set(theme_minimal(base_size = 12))

map_draw_location = here('data', 'map_parameter_draws.RDS')
map_draws = readRDS(map_draw_location)

mcmc_draw_location =  here('data', 'mcmc_parameter_draws.RDS')
mcmc_draws = readRDS(mcmc_draw_location)

#Dose sizes we assess risk over
DD = c(seq(0,10, 0.05),seq(11, 60, 0.5))
nD = length(DD)


# Function to evaluate risk of being below threshold at t=12 post dose.
# This function uses the parameter draws from the posterior to compute the concentration curve
# Then, we can evaluate curves over dose sizes.
# Returns an array of risk which corresponds to dose size
get_risk_at_12 = function(patient, draws, thresh){
  
  n = nrow(draws[[1]])
  #Pk paramters for patient
  ka = rray(draws$ka[,patient], dim=c(n,1))
  ke = rray(draws$ke[,patient], dim=c(n,1))
  cl = rray(draws$cl[,patient], dim=c(n,1))
  time = 12
  
  # Dose sizes to evaluate over
  D = rray(DD, dim = c(1,nD))
  
  #Array broadcasting for economy of thought
  y = 1000*(ke*ka)/(2*cl*(ke-ka))*(exp(-ka*time) - exp(-ke*time))
  yy = y*D
  r= yy %>% rray_lesser(thresh) %>% rray_mean(axes = c(1))
  
  as.numeric(r)
}

# Root finding is used to invert the risk curve
# I provide a risk level, p, it returns a dose size which gives me that risk
estimate_d = function(model, p){
  roots = uniroot(function(x) model(x) - p, c(0, max(DD)))
  roots$root
}

# Determine risk for all patients
risk_at_12<-
  tibble(patient = 1:100) %>% 
  mutate(
    #Get risk for each patient
    map_risk_at_12 = map(patient, ~get_risk_at_12(.x, map_draws, 20)),
    mcmc_risk_at_12 = map(patient, ~get_risk_at_12(.x, mcmc_draws, 20)),
    # Interpolate the risk as a function of dose using a hermite spline
    D = list(DD),
    mcmc_spline = map2(D, mcmc_risk_at_12, ~splinefun(.x, .y, method='hyman')),
    map_spline = map2(D, map_risk_at_12, ~splinefun(.x, .y, method='hyman'))
  )



# Doses for risks p = 0.5 ... 0.95 for MAP and HMC
# Then compute their differences
doses_for_12 = risk_at_12 %>% 
  select(patient, mcmc_spline, map_spline) %>% 
  crossing(p = seq(0.05, 0.95, 0.05)) %>% 
  # This line uses root solving to find the dose required to achieve the desired risk level
  mutate(mcmc_estimated_dose =  map2_dbl(mcmc_spline, p, estimate_d),
         map_estimated_dose =  map2_dbl(map_spline, p, estimate_d),
         delta = map_estimated_dose - mcmc_estimated_dose
  )


# Save dose data to be used later in calibration
doses_for_12%>% 
  select(patient, p, mcmc_estimated_dose, map_estimated_dose) %>% 
  write_csv(here("data","experiment_1_doses.csv"))


figure_7_left = doses_for_12 %>% 
  ggplot(aes(p, delta, group = patient))+
  geom_line()+
  scale_x_continuous(labels = scales::percent, limits = c(0,0.5))+
  scale_y_continuous(limits = c(-3,3))+
  geom_hline(aes(yintercept = 0), color = 'red')+
  labs(x = 'Risk At 12 Hours Post Dose',
       y = 'MAP Dose - HMC Dose')


ggsave('figure_7_left.pdf',
       figure_7_left, 
       path = here("figures"))


# ---- Calibration ----

dose_location = here("data","experiment_1_doses.csv")
doses_for_12%>% 
  select(patient, p, mcmc_estimated_dose, map_estimated_dose) %>% 
  write_csv(dose_location)


# Read in the true pk params
true_pk_param_location = here("data","simulated_data.csv")
true_pk_params = read_csv(true_pk_param_location)

experiment_1_doses = read_csv(here("data","experiment_1_doses.csv")) %>% 
  filter(p<=0.5)


pkfunc<-function(dose, cl, ke, ka, t){
  
  1000*dose*ke*ka/(2*cl*(ke - ka))*(exp(-ka*t) - exp(-ke*t))
  
}


# To determine calibration, we give each patient their recommended dose for the desired risk
# Each dose was designed to elicit a risk of exceeding some threshold
# The calibration is the proportion of those patients who fail to exceed the threshold.
figure_8_left_data = experiment_1_doses %>% 
  left_join(true_pk_params, by = c('patient' = 'subjectids')) %>% 
  mutate(t = 12, 
         conc_mcmc = pkfunc(mcmc_estimated_dose, cl, ke, ka, t),
         conc_map = pkfunc(map_estimated_dose, cl, ke, ka, t),
         calibration_mcmc = conc_mcmc<=20,
         calibration_map = conc_map<=20) %>%
  group_by(p) %>% 
  summarize(mcmc_calib = mean(calibration_mcmc), 
            map_calib = mean(calibration_map))


figure_8_left<-figure_8_left_data %>% 
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

ggsave(filename = 'figure_8_left.pdf',
       plot = figure_8_left,
       path = here('figures'))
