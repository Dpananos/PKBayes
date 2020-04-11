
printf "\n#######################################################################n"
printf "Starting Analysis"
printf "\n#######################################################################n"

printf "\n################\n"
printf "Removing Old Plots"
printf "\n################\n"

rm -v -- figures/*

printf "\n#######################################################################\n"
printf "Drawing From Prior Predictive, Fitting Model, Generating Simulation Data"
printf "\n#######################################################################\n"
Rscript --no-environ R/01_fit_apixaban_model.R

printf "\n#######################################################################\n"
printf "Fitting Simulated Data Using MAP"
printf "\n#######################################################################\n"

Rscript --no-environ R/02_Run_MAP_Fit.R

printf "\n#######################################################################n"
printf "Fitting Simulated Data Using HMC...Be Patient!"
printf "\n#######################################################################n"
Rscript --no-environ R/03_Run_HMC_Fit.R

printf "\n#######################################################################n"
printf "Performing Experiment 1: Risk at 12 Hours"
printf "\n#######################################################################n"
Rscript --no-environ R/04_12_Hour_Calibration.R

printf "\n#######################################################################n"
printf "Performing Experiment 2: Risk at Cmax"
printf "\n#######################################################################n"
Rscript --no-environ R/05_CMax_Calibration.R

printf "\n#######################################################################n"
printf "Making Extra Plots"
printf "\n#######################################################################n"
Rscript --no-environ R/06_Extra_Figures.R