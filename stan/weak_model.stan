functions {
  
  // PK Function.  Solution to differential equation
  // y' = ka*(D/V)exp(-ka*t) - k*y, y(0) = 0
  real PK_profile(real t, real D, real Cl, real k_a, real k) {
    return (D / Cl) * (k_a*k / (k - k_a))
            * ( exp(- k_a * t) - exp(-k * t) );
  }
}
data{
  int N; //Total number of observations
  int subjectids[N]; 
  int n_subjects;
  vector[N] times;
  real yobs[N];
}
parameters{
  real<lower=0> log_CL;
  real<lower=0> s_CL;
  vector[n_subjects] z_CL;
  
  real log_ke;
  real<lower=0> s_ke;
  vector[n_subjects] z_ke;
  
  real<lower=0, upper=1> alpha ;
  real<lower=0> s_ka;
  vector[n_subjects] z_ka;

  real<lower=0, upper=1> delay_mu;
  real<lower=0> delay_kappa;
  vector<lower=0, upper=1>[n_subjects] delays;
  
  real<lower=0> sigma;
  
}
transformed parameters{
  
  real log_ka=log_ke - log(alpha);
  vector[n_subjects] CL = exp(log_CL + z_CL*s_CL);
  vector[n_subjects] ke = exp(log_ke + z_ke*s_ke);
  vector[n_subjects] ka = exp(log_ka + z_ka*s_ka);
  real delay_alpha = delay_mu*delay_kappa;
  real delay_beta = (1-delay_mu)*delay_kappa;
  vector[N] delayed_times = times - 0.5*delays[subjectids];
  vector[N] C;
  
  for (i in 1:N){
    C[i] = PK_profile(delayed_times[i], 5, CL[subjectids[i]], ka[subjectids[i]], ke[subjectids[i]]);
  }
}
model{
  
  log_CL ~ normal(0, 1);
  s_CL ~ cauchy(0,1);
  z_CL ~ normal(0,1);
  
  log_ke ~ normal(0, 1);
  s_ke ~ cauchy(0,1);
  z_ke ~ normal(0,1);
  
  alpha ~ beta(1, 1);
  z_ka ~ normal(0,1);
  s_ka ~ cauchy(0,1);
  
  delay_mu ~ beta(1,1);
  delay_kappa ~ cauchy(0,1);
  delays ~ beta(delay_alpha, delay_beta);
  sigma ~ cauchy(0,1);
  
  yobs ~ lognormal(log(C), sigma);
}
generated quantities{
  
  real z1 = normal_rng(0,1);
  real z2 = normal_rng(0,1);
  real ka_gen = exp(log_ka + z1*s_ka);
  real ke_gen = exp(log_ke + z2*s_ke);
  real tmax = (log(ka_gen) - log(ke_gen))/(ka_gen - ke_gen);
  
}

  