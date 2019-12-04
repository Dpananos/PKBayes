functions {
  real PK_profile(real t, real D, real Cl, real k_a, real k_e) {
    return (D / Cl) * (k_e*k_a / (k_e - k_a))
            * ( exp(- k_a * t) - exp(-k_e * t) );
  }
}
data{
  
  real D;
  
  int N;
  
  int p;
  
  int N_patients;
  
  real time[N];
  
  matrix[N,p] X;
  
  int subjectID[N];
  
  real y[N];
}
parameters{
  // parameter: V Volume
  vector[p] BETA_Cl;
  real<lower=0> SIGMA_Cl;
  vector[N_patients] z_Cl;
  
  // parameter: k Elimination
  vector[p] BETA_ke;
  real<lower=0> SIGMA_ke;
  vector[N_patients] z_ke;
  
  // parameter: ka Absorption
  vector[p] BETA_ka;
  real<lower=0> SIGMA_ka;
  vector[N_patients] z_ka;
  
  //parameter: noise in likelihood
  real<lower=0> sigma;
  
  
}
transformed parameters{
  //Predicted concentrations
  real C[N];
  

  vector[N] Cl;
  vector[N] ke;
  vector[N] ka;

  //Parameter means
  vector[N] MU_KA;
  vector[N] MU_KE;
  vector[N] MU_CL;
  
  MU_KA = X*BETA_ka; //Only Baseline, IsMale, Age,Weight, Creatinine
  MU_KE = X*BETA_ke;
  MU_CL = X*BETA_Cl;  //Only Baseline, Ismale, and Weight
  
  Cl = exp(MU_CL + z_Cl[subjectID]*SIGMA_Cl);
  ke = exp(MU_KE + z_ke[subjectID]*SIGMA_ke);
  ka = exp(MU_KA + z_ka[subjectID]*SIGMA_ka);
 
  for (i in 1:N) C[i] = PK_profile(time[i], D, Cl[i], ka[i], ke[i]);  
  
}
model{
  
  // Priors
  //Coefficients
  BETA_Cl  ~ normal(0,0.1);
  BETA_Cl[1] ~ normal(log(3),0.07);
  BETA_ke  ~ normal(0,0.1);
  BETA_ke[1] ~ normal(log(0.2), 0.1);
  BETA_ka ~ normal(0,0.1);
  
  //Noise
  SIGMA_Cl ~ lognormal(log(0.075), 0.1);
  SIGMA_ke~ lognormal(log(0.2), 0.1);
  SIGMA_ka ~ lognormal(log(0.2), 0.1);
  sigma ~ gamma(2,2);
  
  //Random Effects
  z_Cl ~ normal(0,1);
  z_ke ~ normal(0,1);
  z_ka ~ normal(0,1);

  // Likelihood
 y ~ lognormal(log(C), sigma);
  
}
