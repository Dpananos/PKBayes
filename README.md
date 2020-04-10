
# PKBayes


Code for "Comparisons Between Hamiltonioan Monte Carlo and Maximum A Posteriori For A Bayesian Model For Apixiban Induction Dose & Dose Personalization".


* `analysis/01_apixiban_model.Rmd` has code to generate prior predictive samples and fit the model described in section 3.  Code for figures 1, 3, and 4 are found here.

* `analysis/02_Run_Simulation.Rmd` has code to fit models to pseudopatients.  **Warning**: The model fit via HMC takes approx 1 hour to run, so be prepared to wait.

* `analysis/03_decisions.Rmd` has code to create figures 7 and 8.

* `analysis/04_more_plots.Rmd` has code for figure 6 (and for figure 5, though it is not included).  In order to run this notebook, `analysis/02_Run_Simulation.Rmd` needs to be run first.


