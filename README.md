Research compendium for our article:

<Citation Here>

# Repository Overview

* `data` contains raw data used to fit our apixaban model as well as derived datasets from our modelling pricess.  The `data/README` provides a codebook for the data in each file.

* `analysis` contains the scripts that fit the models, compute appropriate metrics, and produce figures.  The `analysis/README` outlines what datasets and what figures are produced by which scripts.

* `figures` contains figures generated for the paper.

* `models` contains relevant Stan models used in the paper.  The `models/README` includes a summary of the models and in which scripts they are used.

The shell script `build.sh` will delete all figures and then rerun the scripts in `analysis` in order to generate them again.  **Fitting our simulation via Hamiltonian Monte Carlo can take up to an hour on a 2017 iMac with 8GB of RAM**.