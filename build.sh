echo "Starting Analysis"

echo "Removing Old Plots"

rm -v -- figures/*

echo "Drawing From Prior Predictive, Fitting Model, Generating Simulation Data"
Rscript --no-environ R/01_fit_apixaban_model.R

echo "Fitting Simulated Data Using MAP"
Rscript --no-environ R/02_Run_MAP_Fit.R

echo "Fitting Simulated Data Using HMC...Be Patient!"
Rscript --no-environ R/03_Run_HMC_Fit.R

echo "Performing Experiment 1: Risk at 12 Hours"
Rscript --no-environ R/04_12_Hour_Calibration.R

echo "Performing Experiment 2: Risk at Cmax"
Rscript --no-environ R/05_CMax_Calibration.R

echo "Making Extra Plots"
Rscript --no-environ R/06_Extra_Figures.R