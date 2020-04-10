
data {
   int N;
   vector[N] t;
   
   //----tmax hyperiors;
   
   real MU_T_MEAN;
   real MU_T_SIGMA;
   
   real S_T_A;
   real S_T_B;
   
   real MU_CL_MEAN;
   real MU_CL_SIGMA;
   
   real S_CL_A;
   real S_CL_B;
   
   real PHI_A;
   real PHI_B;
    
   real KAPPA_A;
   real KAPPA_B;
   
   real SIGMA_MEAN;
   real SIGMA_SIGMA;
   
}
generated quantities{
  
  real mu_t = normal_rng( MU_T_MEAN, MU_T_SIGMA);
  real s_t = gamma_rng(S_T_A, S_T_B);
  real tmax = exp(mu_t + normal_rng(0,1)*s_t);
  
  real alpha=beta_rng(2,2);
  
  real ka = log(alpha)/(tmax * (alpha-1));
  real ke = alpha*ka;
  
  real mu_Cl = lognormal_rng(MU_CL_MEAN, MU_CL_SIGMA);
  real s_Cl = gamma_rng(S_CL_A, S_CL_B);
  real CL = exp(mu_Cl + normal_rng(0,1)*s_Cl);
  
  real phi = beta_rng(PHI_A, PHI_B);
  real kappa = beta_rng(KAPPA_A, KAPPA_B);
  real delay = beta_rng(phi/kappa, (1-phi)/kappa);
  
  vector[N] y=2.5/CL*(ke*ka)/(ke-ka)*(exp(-ka*(t-0.5*delay)) - exp(-ke*(t-0.5*delay)));
  
  real sigma = lognormal_rng(SIGMA_MEAN, SIGMA_SIGMA);
  
  vector[N] Observations;
  
  for (i in 1:N){
     Observations[i] = lognormal_rng(log(y[i]), sigma);
  }

}
