import numpy as np 
import pickle
import arviz as az
import pandas as pd
import pymc3 as pm
import argparse
from scipy.stats import norm, binom, uniform

from .regression_models import strong_regression_model_factory


def save_obj(obj, name):
    with open(f'{name}.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(f'{name}.pkl', 'rb') as f:
        return pickle.load(f)


def generate_times_subjects(t, subjects, random_sample = False):
    #TODO: Mix and match version.
    '''Function to return times and subject Ids for convienience.  
    
    Inputs:
    t -- array of desired observation times or int for number of observation times between 0.5 and 12
    subjects -- integer for number of subjects to create
    
    '''

    if isinstance(t, int) and not random_sample:
        times = np.tile(np.linspace(0.5,12, t), subjects)
        subject_ids = np.concatenate([t*[j] for j in range(subjects)])
        
    elif isinstance(t, int) and random_sample:
        
        subject_ids = np.concatenate([t*[j] for j in range(subjects)])
        times = uniform(0.5,12).rvs(size=len(subject_ids), random_state = 19920908)
        

    else:
        times = np.tile(t, subjects)
        subject_ids = np.concatenate([len(t)*[j] for j in range(subjects)])

    return times, subject_ids

def generate_data(t, subjects, random_sample):
    
    times, subject_ids = generate_times_subjects(t, subjects, random_sample)
    
    n_subjects = len(np.unique(subject_ids))
    Xc = norm().rvs(size=(n_subjects,2), random_state = 19920908)
    Xb = binom(n=1,p=0.5).rvs(size=(n_subjects), random_state = 19920908)
    X = np.c_[Xb, Xc]

    # generate some regression data too
    with strong_regression_model_factory(None, X, times, times, subject_ids, subject_ids, use_delay=True):
        regression_bootstrap_prior = pm.sample_prior_predictive(1, random_seed=19920908)
    
    regression_bootstrap_prior['times'] = times
    regression_bootstrap_prior['subject_ids'] = subject_ids
    regression_bootstrap_prior['X'] = X
    save_obj(regression_bootstrap_prior, 'data/regression_bootstrap_data')
    print('Data Generated!')

    
    
    
def summarize_posterior(data, var_name, model_name):
    
    pred = (
            data[var_name].
            to_dataframe().
            groupby(level=2).  #This should be the dimension
            agg([('pred',np.mean), 
                ('low', lambda x: np.quantile(x,0.025)),
                ('high', lambda x: np.quantile(x,0.975)) ])
        )

    pred.columns = pred.columns.droplevel()
    pred.index.name = None
    pred.columns = [f'{model_name}_{j}' for j in pred.columns]

    return pred

def summarize_ppc(data, var_name, model_name):
    
    pred = (
            data[var_name].
            to_dataframe().
            groupby(level=2).  #This should be the dimension
            agg([('pred',np.mean), 
                ('low', lambda x: np.quantile(x,0.025)),
                ('high', lambda x: np.quantile(x,0.975)) ])
        )

    pred.columns = pred.columns.droplevel()
    pred.index.name = None
    pred.columns = [f'{model_name}_{j}' for j in pred.columns]

    return pred


