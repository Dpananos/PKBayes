import pickle
import pystan
import numpy as np
import argparse
import functools
from utils.stan_tools import *
from utils.python_tools import *


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate data to fit the model")
    parser.add_argument("--seed", type=int, default=19920908)
    parser.add_argument("--n_subjects", type=int, default=36)
    parser.add_argument("--spacing", type=str, default="even", choices=["even", "random"])
    parser.add_argument("--tmin", type=float, default=0.5)
    parser.add_argument("--tmax", type=float, default=12)
    parser.add_argument("--n_obs", type=int, default=8)

    parser.add_argument("--D", type=int, default=5)
    parser.add_argument("--n_continuous_predictors", type=int, default=2)
    parser.add_argument("--n_binary_predictors", type=int, default=2)
    parser.add_argument("--binary_prob", type=float, default=0.33)

    parser.add_argument('--alpha', type=float,  default = 0.5)
    parser.add_argument("--beta_cov", nargs=3, default=[0.075, 0.075, 0.075])
    parser.add_argument("--rfx_cov", nargs=3, default=[0.07, 0.2, 0.2])
    parser.add_argument("--use_normal_rfx", type=int, default=1)
    parser.add_argument("--sigma_obs", type=float, default=0.12)
    parser.add_argument("--use_delay", type=bool, default=True)
    args = parser.parse_args()

    # Create some names for our patients.  Zero pad to keep them to easily order them.  Comes in handy when we compare
    # Random effects etc.  Else, subejcts may be ordered like subject_1, subject_10, subject_100, etc.
    subjects = np.sort([f"subject_{j:04}" for j in range(args.n_subjects)] * args.n_obs)

    # Create times for observations
    if args.spacing == "even":
        # If even spacing, then observe each subject at the same time
        obs_times = np.linspace(args.tmin, args.tmax, args.n_obs)
        times = np.tile(obs_times, args.n_subjects)
    else:
        # If we don't observe at the same time, then observation times are uniformly random between tmin and tmax
        obs_times = [np.sort(np.random.uniform(low=args.tmin, high=args.tmax, size=args.n_obs)) for j in range(args.n_subjects)]
        # Store the times in a flat list
        times = list(np.concatenate(obs_times))

    # See utils/pthon_tools.py
    X = generate_covariates(args.n_subjects, args.n_continuous_predictors,
    args.n_binary_predictors, args.binary_prob)
    betas = generate_regression_coefficients(X, args.beta_cov)
    # Baseline variables are Cl, ke, ka
    # Note that ka>ke.  The non-dimensionalization y = D/Cl*C and tau = ka*t yields a nd constant
    # between 0 and 1 (alpha = ke/ka), or alternatively alpha>1 if alpha = ka/ke.  
    # On the log space, multuplication is addition
    # so ka=(1/alpha)*ke  ==> exp(log(ka)) = exp(log(1/alpha*ke)) = exp(log(1/alpha) + log(ke))
    beta_baseline = [np.log(3.3), np.log(1/args.alpha) + np.log(0.5), np.log(0.5)] 
    pk = generate_pk_params(X, betas, beta_baseline, args.use_normal_rfx, args.rfx_cov)

    # Observational model is lognormal
    # PK function is y' = (D/Cl)*ke*ka*exp(-ka*t) - ke*y
    df = make_obs(args.D, subjects, times, X, pk, args.sigma_obs, use_delay=args.use_delay)
    df.to_csv("data/test.csv")

    with open('data/reg_coefs.txt', 'wb') as f:
        # Save for comparison later.
        pickle.dump({'betas':betas}, f, protocol=pickle.HIGHEST_PROTOCOL)
