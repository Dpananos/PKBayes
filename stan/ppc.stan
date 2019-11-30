functions {
  
  // PK Function.  Solution to differential equation
  // y' = ka*(D/V)exp(-ka*t) - k*y, y(0) = 0
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
generated quantities{
vector[p] BETA_Cl = to_vector(normal_rng(rep_vector(0,p), rep_vector(0.1,p)));
vector[p] BETA_ke = to_vector(normal_rng(rep_vector(0,p), rep_vector(0.1,p)));
vector[p] BETA_ka = to_vector(normal_rng(rep_vector(0,p), rep_vector(0.1,p)));

real SIGMA_Cl = lognormal_rng(log(.075),.1);
real SIGMA_ke = lognormal_rng(log(.2),.1);
real SIGMA_ka = lognormal_rng(log(.2),.1);
real sigma = gamma_rng(2,2);

vector[N_patients] z_Cl = to_vector(normal_rng(rep_vector(0,N_patients), rep_vector(1,N_patients)));
vector[N_patients] z_ka = to_vector(normal_rng(rep_vector(0,N_patients), rep_vector(1,N_patients)));
vector[N_patients] z_ke = to_vector(normal_rng(rep_vector(0,N_patients), rep_vector(1,N_patients)));

vector[N] Cl = exp(X*BETA_Cl + z_Cl[subjectID]*SIGMA_Cl);
vector[N] ke = exp(X*BETA_ke + z_ke[subjectID]*SIGMA_ke);
vector[N] ka = exp(X*BETA_ka + z_ka[subjectID]*SIGMA_ka);

vector[N] C;
vector[N] C_pred;

BETA_Cl[1] = normal_rng(log(3),0.07);
BETA_ke[1] = normal_rng(log(1),0.2);
BETA_ka[1] = normal_rng(log(0.2),0.2);

for(i in 1:N) {
  C[i] = PK_profile(time[i], D, Cl[i], ka[i], ke[i]);
  C_pred[i] = lognormal_rng(log(C[i]), sigma);
}  

}
