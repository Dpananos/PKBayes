data{
  int N; //Total number of observations
  int subjectids[N]; 
  int n_subjects;
  vector[N] times;
  real yobs[N];
  
  
  
}
parameters{
  real<lower=0>  mu_CL;
  real<lower=0> s_CL;
  vector[n_subjects] z_CL;
  
  real<lower=0> mu_t;
  real<lower=0> s_t;
  vector[n_subjects] z_t;
  
  vector<lower=0, upper=1>[n_subjects] alpha ;
  
  real<lower=0, upper=1> phi;
  real<lower=0, upper=1> kappa;
  vector<lower=0, upper=1>[n_subjects] delays;
  
  real<lower=0> sigma;
  
  
  
}
transformed parameters{
  vector[n_subjects] Cl = exp(mu_CL + z_CL*s_CL);
  vector[n_subjects] t = exp(mu_t + z_t*s_t);
  vector[n_subjects] ka = log(alpha)./(t .* (alpha-1));
  vector[n_subjects] ke = alpha .* ka;
  vector[N] delayed_times = times - 0.5*delays[subjectids];
  
  vector[N] C = (2.5 ./ Cl[subjectids]) .* (ke[subjectids] .* ka[subjectids]) ./ (ke[subjectids] - ka[subjectids]) .* (exp(-ka[subjectids] .* delayed_times) -exp(-ke[subjectids] .* delayed_times));
}
model{
  mu_CL ~ gamma(45,50);
  s_CL ~ gamma(15,100);
  z_CL ~ normal(0,1);
  
  mu_t ~ normal(log(3.3), 0.25);
  s_t ~ gamma(15, 100);
  z_t ~ normal(0,1);
  
  alpha ~ beta(2,2);
  phi ~ beta(1,1);
  kappa ~ beta(5,5);
  delays ~ beta(phi/kappa, (1-phi)/kappa);
  sigma ~ lognormal(log(0.1), 0.2);
  yobs ~ lognormal(log(C), sigma);
}
