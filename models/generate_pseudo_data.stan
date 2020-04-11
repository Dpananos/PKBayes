data{
  int N_subjects;
  int N;
  int subjectids[N];
  vector[N] times;
}
generated quantities{
  //All we are doing here is generating random numbers.
  //This could have been done in R, but it helps to write it down in Stan to
  //have a concrete idea of what the model is doing.
  real mu_CL = normal_rng(1.64, 0.09);
  real s_CL = lognormal_rng(-0.97, 0.11);
  real sigma = lognormal_rng(-1.76, 0.063);
  real mu_t = normal_rng(0.97, 0.05);
  real s_t = lognormal_rng(-1.42, 0.12);
  
  
  vector[N_subjects] z_CL;
  vector[N_subjects] z_t;
  vector[N_subjects] alpha;
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
  }
 
   tmax = exp(mu_t + z_t*s_t);
   Cl = exp(mu_CL + z_CL * s_CL);
   ka = log(alpha) ./ (tmax .* (alpha-1));
   ke = alpha .* ka;
  
   C = (2.5 ./ Cl[subjectids]) .* 
   (ke[subjectids] .* ka[subjectids]) ./ (ke[subjectids] - ka[subjectids]) .* 
   (exp(-ka[subjectids] .* delayed_times) -exp(-ke[subjectids] .* delayed_times)); 
   
   for (i in 1:N){
     Cobs[i] = lognormal_rng(log(C[i]), sigma);
   }
}
