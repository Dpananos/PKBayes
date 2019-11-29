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
    parser.add_argument(
        "--spacing", type=str, default="even", choices=["even", "random"]
    )
    parser.add_argument("--tmin", type=float, default=0.5)
    parser.add_argument("--tmax", type=float, default=12)
    parser.add_argument("--n_obs", type=int, default=8)

    parser.add_argument("--D", type=int, default=5)
    parser.add_argument("--n_continuous_predictors", type=int, default=2)
    parser.add_argument("--n_binary_predictors", type=int, default=2)
    parser.add_argument("--binary_prob", type=float, default=0.33)
    parser.add_argument("--beta_baseline", nargs=3, default=[np.log(3.3), np.log(0.2), np.log(1)])
    parser.add_argument("--beta_cov", nargs=3, default=[0.075, 0.075, 0.075])
    parser.add_argument("--rfx_cov", nargs=3, default=[0.07, 0.2, 0.2])
    parser.add_argument("--use_normal_rfx", type=int, default=1)
    parser.add_argument("--sigma_obs", type=float, default=0.12)
    args = parser.parse_args()

    subjects = np.sort([f"subject_{j:04}" for j in range(args.n_subjects)] * args.n_obs)

    # Create times for observations
    if args.spacing == "even":
        # If even spacing, then observe each subject at the same time
        obs_times = np.linspace(args.tmin, args.tmax, args.n_obs)
        times = np.tile(obs_times, args.n_subjects)
    else:
        obs_times = [
            np.sort(np.random.uniform(low=args.tmin, high=args.tmax, size=args.n_obs))
            for j in range(args.n_subjects)
        ]
        times = functools.reduce(lambda x, y: x + y, obs_times)

    data_params = {
        "n_subjects": args.n_subjects,
        "n_continuous_predictors": args.n_continuous_predictors,
        "n_binary_predictors": args.n_binary_predictors,
        "binary_prob": args.binary_prob,
        "beta_baseline": args.beta_baseline,
        "beta_cov": args.beta_cov,
        "rfx_cov": args.rfx_cov,
        "use_normal_rfx": args.use_normal_rfx
    }

    model = StanModel_cache(file="stan/data_gen.stan")
    fit = model.sampling(data=data_params, algorithm="Fixed_param", iter=1, chains=1, seed=args.seed).extract()

    pk = np.squeeze(fit["pk"])
    X = np.squeeze(fit["X"])
    df = make_obs(args.D, subjects, times, X, pk, args.sigma_obs)

    df.to_csv("data/test.csv")
