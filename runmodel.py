import pystan 
import numpy as np 
import pandas as pd
from utils.python_tools import *
from utils.stan_tools import *

df = pd.read_csv('data/test.csv', index_col = 0).reset_index()
df['subject'] = df['index'].astype('category').cat.codes+1

sm = StanModel_cache('stan/model.stan')

X = df.iloc[:,df.columns.str.contains('x_')].values
y = df.y_obs.values
N = y.size
p = X.shape[1]
time = df.t.values
subject = df.subject.values


d = {
    'D':5,
    'N':N,
    'p':p,
    'N_patients': np.unique(subject).size,
    'time':time,
    'X':X,
    'subjectID':subject,
    'y':y
    
}

print(sm)

fit = sm.sampling(data = d, chains = 4)

print(pystan.check_hmc_diagnostics(fit))