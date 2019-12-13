functions {
  real PK_profile(real t, real D, real Cl, real k_a, real k_e) {
    return (D / Cl) * (k_e*k_a / (k_e - k_a))
            * ( exp(- k_a * t) - exp(-k_e * t) );
  }
}
data{
  
  real D;
  
  int N; //len of whole shebang
  
  int p;
  
  int n;
  
  real time[N];
  
  matrix[n,p] X;
  
  int subjectID[N];
  
  real y[N];
}
parameters{
  // parameter: V Volume
  real<lower=0> baseline_cl;
  vector[p] BETA_Cl;
  real<lower=0> SIGMA_Cl;
  vector[n] z_Cl;
  
  // parameter: k Elimination
  real baseline_ke;
  vector[p] BETA_ke;
  real<lower=0> SIGMA_ke;
  vector[n] z_ke;
  
  // parameter: ka Absorption
  real<lower=baseline_ke> baseline_ka;
  vector[p] BETA_ka;
  real<lower=0> SIGMA_ka;
  vector[n] z_ka;
  
  //parameter: noise in likelihood
  real<lower=0> sigma;


  real<lower=0, upper=1> mu;
  real<lower=0> kappa;
  real<lower=0, upper=1> delay[n];
  
  
}
transformed parameters{
  //Predicted concentrations
  real C[N];
  
  vector[n] Cl;
  vector[n] ke;
  vector[n] ka;

  Cl = exp(baseline_cl + X*BETA_Cl + z_Cl*SIGMA_Cl);
  ke = exp(baseline_ke + X*BETA_ke + z_ke*SIGMA_ke);
  ka = exp(baseline_ka + X*BETA_ka + z_ka*SIGMA_ka);
 
  for (i in 1:N) C[i] = PK_profile(time[i] - 0.5*delay[subjectID[i]], D, Cl[subjectID[i]], ka[subjectID[i]], ke[subjectID[i]]);  
  
}
model{
  
  // Priors
  //Coefficients
  baseline_cl ~ lognormal(log(3),1);
  baseline_ke ~ normal(0,1);
  baseline_ke ~ normal(0,1);

  BETA_Cl  ~ normal(0,0.1);
  BETA_ke  ~ normal(0,0.1);
  BETA_ka ~ normal(0,0.1);
  
  //Noise
  SIGMA_Cl ~ cauchy(0,1);
  SIGMA_ke~ cauchy(0,1);
  SIGMA_ka ~ cauchy(0,1);
  sigma ~ cauchy(0,1);
  
  //Random Effects
  z_Cl ~ normal(0,1);
  z_ke ~ normal(0,1);
  z_ka ~ normal(0,1);

  mu ~ beta(10,10);
  kappa ~ pareto(10,1.5);
  delay ~ beta(mu*kappa, (1-mu)*kappa);

  // Likelihood
  for (i in 1:N){target += lognormal_lpdf(y[i] | log(C[i]), sigma);}
  
}
