
regression_posterior = pk_regression(df, draws=1000, tune=1000, chains=8, random_seed=19920908, target_accept = 0.95)

az.to_netcdf(regression_posterior, "trace/pk_regression.NC")