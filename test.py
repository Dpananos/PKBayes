import pandas as pd
from pymc3_models import *
import arviz as az
import matplotlib.pyplot as plt
import os

df = pd.read_csv("data/train.csv", index_col=0)

regression_posterior = pk_regression(df, draws=1000, tune=1000, chains=12, random_seed=19920908, target_accept = 0.95)

az.to_netcdf(regression_posterior, "trace/pk_regression.NC")

posterior = pk_mixed_model(df, draws=1000, tune=1000, chains=12, random_seed=19920908, target_accept=0.95)

az.to_netcdf(posterior, "trace/pk_mixed_model.NC")

