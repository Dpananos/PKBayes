import numpy as np
import arviz as az
import pandas as pd
import pymc3 as pm
import argparse

from utils.models import strong_model_factory
from utils.regression_models import strong_regression_model_factory
from utils.tools import generate_times_subjects
from utils.tools import  save_obj

# Simulate from the prior predictive to get some data to fit
t = 2
subjects = 600
times, subject_ids = generate_times_subjects(t, subjects, random_sample = True)

# A model to generate some data.  Not actually bootstrapping anything
with strong_model_factory(None, times, subject_ids, use_delay = True):    
    bootstrap_prior = pm.sample_prior_predictive(1, random_seed=19920908)

# generate some regression data too
with strong_regression_model_factory(None, times, subject_ids):
    regression_bootstrap_prior = pm.sample_prior_predictive(1, random_seed=19920908)
    

bootstrap_prior['times'] = times
bootstrap_prior['subject_ids'] = subject_ids
save_obj(bootstrap_prior, 'data/bootstrap_data')

regression_bootstrap_prior['times'] = times
regression_bootstrap_prior['subject_ids'] = subject_ids
save_obj(regression_bootstrap_prior, 'data/regression_bootstrap_data')
