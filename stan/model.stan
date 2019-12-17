

functions {
  real PK_profile(real t, real D, real Cl, real k_a, real k_e) {
    return (D / Cl) * (k_a*k_e / (k_a - k_e))
            * ( exp(- k_a * t) - exp(-k_e * t) );
  }
}
data{
  
  //Total Dose in mg
  real D;
  
  int N;
  
  int N_subjects;
  
  real time[N];
  
  int subjectID[N];
  
  real yobs[N];
}
parameters{
  // parameter: V Volume
  real<lower=0> baseline_cl;
  real<lower=0> SIGMA_cl;
  vector[N_subjects] z_cl;
  
  // parameter: k Elimination
  real baseline_ke;
  real<lower=0> SIGMA_ke;
  vector[N_subjects] z_ke;
  
  // parameter: ka Absorption
  real<lower=baseline_ke> baseline_ka;
  real<lower=0> SIGMA_ka;
  vector[N_subjects] z_ka;
  
  //parameter: noise in likelihood
  real<lower=0> sigma;
  
  //Delay effects
  //See "Modeling of delays in PKPD: classical approaches and 
  //a tutorial for delay differential equations" -- Koch G., et. al
  //From a ODE model, we can cook up a delay.
  //Delay, in this context, is an estimate of the mean transit time.
  //Hence, times are interpreted as times after absorption, 
  //rather than times after administration
  
  real<lower=0,upper=1> phi; //Use to measure population delay.
  real<lower=10> lambda;
  vector<lower=0, upper=1>[N_subjects] delay_raw; //Each patient has their own delay
}
transformed parameters{
  //Predicted concentrations
  real C[N];
  
  //Parameters for each patient
  vector[N_subjects] k_a;
  vector[N_subjects] k_e;
  vector[N_subjects] Cl;
  vector[N] delta;
  
  Cl = exp(baseline_cl + z_cl*SIGMA_cl);
  k_e = exp(baseline_ke + z_ke*SIGMA_ke);
  k_a = exp(baseline_ka + z_ka*SIGMA_ka);
  delta = 0.5*delay_raw[subjectID];
  for (i in 1:N){
      //Our pharmacologist says that there can be a delay 
      //in the absorption of the drug
      //So although the ODE model assumes the drug is instantaneously absorbed, 
      //this is def not the case.
      //We posit that the drus is absorbed a little later than we think, 
      //which translates to an error in our time measurements
      //Alternatively, that the times are times after absorption, 
      //not times after administration.
    C[i] = PK_profile(time[i] - delta[i], D, Cl[subjectID[i]], k_a[subjectID[i]], k_e[subjectID[i]]);  
  }
  
}
model{
  
  // Priors
  //Coefficients
  baseline_cl  ~ normal(log(3),0.5);
  baseline_ke ~ normal(0,1);
  baseline_ka ~ normal(0,1);
  
  //Noise
  SIGMA_cl ~ cauchy(0,2);
  SIGMA_ke ~ cauchy(0,2);
  SIGMA_ka ~cauchy(0,2);
  sigma ~ cauchy(0,2);
  
  //Random Effects
  z_cl ~ normal(0,1);
  z_ke ~ normal(0,1);
  z_ka ~ normal(-1,1);

  
  //Delay
  //These priors make more sense to me and
  //result in posteriors which are not much different.
  
  //Phi is the mean of the betra
  //lambda sum of successes and losses
  //See stan manual See section 20.2 : Reparameteriztions
  phi ~ beta(2.5, 2.5);
  lambda ~ pareto(10, 1.5);
  delay_raw ~ beta(lambda * phi, lambda * (1 - phi));

  // Likelihood
  yobs ~ normal(C, sigma);
  
}
generated quantities{
  real C_ppc[N];
  vector[N] log_lik;
  
  C_ppc = lognormal_rng(log(C), sigma);
  
  for (i in 1:N)
    log_lik[i] = lognormal_lpdf(yobs[i] | log(C[i]), sigma);
}
