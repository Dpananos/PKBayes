data{
  int N_subjects;
  int N;
  int subjectids[N];
  vector[N] times;

  
}
generated quantities{
  real mu_CL = lognormal_rng(0.26, 0.042);
  real s_CL = lognormal_rng(-1.14, 0.09);
  // real phi = beta_rng(43.98, 39.47);
  // real kappa = beta_rng(10.41, 9.5);
  real sigma = lognormal_rng(-1.76, 0.063);
  real mu_t = lognormal_rng(-0.028, 0.051);
  real s_t = lognormal_rng(-1.40, 0.12);
  
  
  vector[N_subjects] z_CL;
  vector[N_subjects] z_t;
  vector[N_subjects] alpha;
  // vector[N_subjects] delays;
  vector[N_subjects] ke;
  vector[N_subjects] ka;
  vector[N_subjects] Cl;
  vector[N_subjects] tmax;
  vector[N] delayed_times = times;
  vector[N] C;
  vector[N] Cobs;
  
  for (i in 1:N_subjects){
    z_CL[i] = normal_rng(0,1);
    z_t[i] = normal_rng(0,1);
    alpha[i] = beta_rng(2,2);
    // delays[i] = beta_rng(phi/kappa, (1-phi)/kappa);
    
  }
 
   tmax = exp(mu_t + z_t*s_t);
   Cl = exp(mu_CL + z_CL * s_CL);
   ka = log(alpha) ./ (tmax .* (alpha-1));
   ke = alpha .* ka;
   
   // delayed_times = times - 0.5*delays[subjectids];
   
   C = (2.5 ./ Cl[subjectids]) .* 
   (ke[subjectids] .* ka[subjectids]) ./ (ke[subjectids] - ka[subjectids]) .* 
   (exp(-ka[subjectids] .* delayed_times) -exp(-ke[subjectids] .* delayed_times)); 
   
   
   for (i in 1:N){
     Cobs[i] = lognormal_rng(log(C[i]), sigma);
   }
}
