import pandas as pd
from pymc3_models import *
import arviz as az
import matplotlib.pyplot as plt


df = pd.read_csv('data/test.csv', index_col = 0)

posterior = pk_mixed_model(df, random_seed=0)

az.plot_density(posterior, var_names=['alpha'])
plt.show()
