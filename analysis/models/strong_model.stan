
data{
  int N; //Total number of observations
  int subjectids[N]; 
  int n_subjects;
  vector[N] times;
  real yobs[N];
  
  int Ntest;
  vector[Ntest] test_times;
  int test_ids[Ntest];
}
parameters{
  real<lower=0>  mu_CL;
  real<lower=0> s_CL;
  vector[n_subjects] z_CL;
  
  real<lower=0> mu_t;
  real<lower=0> s_t;
  vector[n_subjects] z_t;
  
  vector<lower=0, upper=1>[n_subjects] alpha ;
  
  real<lower=0> sigma;
  
}
transformed parameters{
  vector[n_subjects] Cl = exp(mu_CL + z_CL*s_CL);
  vector[n_subjects] t = exp(mu_t + z_t*s_t);
  vector[n_subjects] ka = log(alpha)./(t .* (alpha-1));
  vector[n_subjects] ke = alpha .* ka;
  
  vector[N] C = (2.5 ./ Cl[subjectids]) .* (ke[subjectids] .* ka[subjectids]) ./ (ke[subjectids] - ka[subjectids]) .* (exp(-ka[subjectids] .* times) -exp(-ke[subjectids] .* times));
}
model{
  mu_CL ~ normal(1.64,0.09);
  s_CL ~ lognormal(-0.94,0.11);
  z_CL ~ normal(0,1);
  
  mu_t ~ normal(0.97, 0.05);
  s_t ~ lognormal(-1.42, 0.12);
  z_t ~ normal(0,1);
  
  alpha ~ beta(2,2);
  sigma ~ lognormal(-1.76, 0.063);
  yobs ~ lognormal(log(C), sigma);
}
generated quantities{
  vector[Ntest] ypred;
  vector[Ntest] yppc;
  
  ypred = (2.5 ./ Cl[test_ids]) .* (ke[test_ids] .* ka[test_ids]) ./ (ke[test_ids] - ka[test_ids]) .* (exp(-ka[test_ids] .* test_times) -exp(-ke[test_ids] .* test_times));
  
  for (i in 1:Ntest){
    yppc[i] = lognormal_rng(log(ypred[i]), sigma);
  }
  
  
  
}
