import numpy as np
import pymc3 as pm
import theano.tensor as tt
import scipy

#TODO: Non-dimensionalization should be done after regression and not only on the intercept.

def strong_regression_model_factory(Yobs, times, subject_ids):

    """Returns a model context for a strongly informative regression model.
    
    Inputs:
    Yobs -- array of observed concentrations at indicated times
    times -- array of observation tiems
    subject_ids -- array of integers used as subject identifiers

    Returns:
    pk_model -- Model context.

    """

    # This is the case when I am simulating from the prior predictive.
    if Yobs is None:
        Yobs = np.zeros_like(times)

    # Generate covariates for the subjects
    eigs = np.array([0.01, 0.24, 0.25, 0.5])
    eigs*=eigs.size
    S = scipy.stats.random_correlation.rvs(eigs, random_state = 19920908)

    # Draw the covariates
    n_subjects = len(np.unique(subject_ids))
    Xs = scipy.stats.multivariate_normal(mean = np.zeros(eigs.size), cov = S).rvs(size=n_subjects, random_state = 0)

    X = Xs[np.unique(subject_ids)]


    with pm.Model() as pk_model:

        log_CL = pm.Normal("log_CL", tt.log(3.5), 0.075)
        betas_CL = pm.Normal('betas_CL', mu=0, sigma=0.12, shape=X.shape[1])
        z_CL = pm.Normal("z_CL", 0, 1, shape=len(np.unique(subject_ids)))
        s_CL = pm.Lognormal("s_CL", np.log(0.12), 0.05)

        log_ke = pm.Normal("log_ke", -1.5, 0.05)
        betas_ke = pm.Normal('betas_ke', mu=0, sigma=0.12, shape=X.shape[1])
        z_ke = pm.Normal("z_ke", 0, 1, shape=len(np.unique(subject_ids)))
        s_ke = pm.Lognormal("s_ke", tt.log(0.12), 0.05)

        alpha = pm.Beta("alpha", 300, 600)
        log_ka = pm.Deterministic("log_ka", log_ke - tt.log(alpha))
        betas_ka = pm.Normal('betas_ka', mu=0, sigma=0.12, shape=X.shape[1])
        z_ka = pm.Normal("z_ka", 0, 1, shape=len(np.unique(subject_ids)))
        s_ka = pm.Lognormal("s_ka", tt.log(0.12), 0.05)

        CL = pm.Deterministic("Cl", tt.exp(log_CL + pm.math.dot(X,betas_CL) + z_CL[np.unique(subject_ids)] * s_CL))

        ke = pm.Deterministic("ke", tt.exp(log_ke + pm.math.dot(X,betas_ke) + z_ke[np.unique(subject_ids)] * s_ke))

        ka = pm.Deterministic("ka", tt.exp(log_ka + pm.math.dot(X,betas_ka) + z_ka[np.unique(subject_ids)] * s_ka))

        y_est = (
            5
            / CL[subject_ids]
            * (ke[subject_ids] * ka[subject_ids])
            / (ke[subject_ids] - ka[subject_ids])
            * (tt.exp(-ka[subject_ids] * times) - tt.exp(-ke[subject_ids] * times))
        )

        y_conc = pm.Deterministic("y_est", y_est)
        sigma = pm.Lognormal("sigma", tt.log(0.12), 0.1)

        y = pm.Lognormal("Yobs", tt.log(y_est), sigma, observed=Yobs)

    return pk_model


def weak_regression_model_factory(Yobs, times, subject_ids):
    """Returns a model context for a weakly informative regression model.

    Inputs:
    Yobs -- array of observed concentrations at indicated times
    times -- array of observation tiems
    subject_ids -- array of integers used as subject identifiers

    Returns:
    pk_model -- Model context.

    """

    # This is the case when I am simulating from the prior predictive.
    if Yobs is None:
        Yobs = np.zeros_like(times)

    # Generate covariates for the subjects
    eigs = np.array([0.01, 0.24, 0.25, 0.5])
    eigs*=eigs.size
    S = scipy.stats.random_correlation.rvs(eigs, random_state = 19920908)

    # Draw the covariates
    n_subjects = len(np.unique(subject_ids))
    Xs = scipy.stats.multivariate_normal(mean = np.zeros(eigs.size), cov = S).rvs(size=n_subjects, random_state = 0)

    X = Xs[np.unique(subject_ids),:]

    with pm.Model() as pk_model:

        log_CL = pm.Bound(pm.Normal, lower=0)("log_CL", tt.log(3.5), 1)
        betas_CL = pm.Normal('betas_CL', mu=0, sigma=0.12, shape=X.shape[1])
        z_CL = pm.Normal("z_CL", 0, 1, shape=len(np.unique(subject_ids)))
        s_CL = pm.HalfCauchy("s_CL", 1)

        log_ke = pm.Normal("log_ke", -1.5, 1)
        betas_ke = pm.Normal('betas_ke', mu=0, sigma=0.12, shape=X.shape[1])
        z_ke = pm.Normal("z_ke", 0, 1, shape=len(np.unique(subject_ids)))
        s_ke = pm.HalfCauchy("s_ke", 1)

        alpha = pm.Beta("alpha", 1, 1)
        log_ka = pm.Deterministic("log_ka", log_ke - tt.log(alpha))
        betas_ka = pm.Normal('betas_ka', mu=0, sigma=0.12, shape=X.shape[1])
        z_ka = pm.Normal("z_ka", 0, 1, shape=len(np.unique(subject_ids)))
        s_ka = pm.HalfCauchy("s_ka", 1)

        CL = pm.Deterministic("Cl", tt.exp(log_CL + pm.math.dot(X,betas_CL) + z_CL[np.unique(subject_ids)] * s_CL))

        ke = pm.Deterministic("ke", tt.exp(log_ke + pm.math.dot(X,betas_ke) + z_ke[np.unique(subject_ids)] * s_ke))

        ka = pm.Deterministic("ka", tt.exp(log_ka + pm.math.dot(X,betas_ka) + z_ka[np.unique(subject_ids)] * s_ka))


        y_est = (
            5
            / CL[subject_ids]
            * (ke[subject_ids] * ka[subject_ids])
            / (ke[subject_ids] - ka[subject_ids])
            * (tt.exp(-ka[subject_ids] * times) - tt.exp(-ke[subject_ids] * times))
        )

        y_conc = pm.Deterministic("y_est", y_est)
        sigma = pm.HalfCauchy("sigma", 1)

        y = pm.Lognormal("Yobs", tt.log(y_est), sigma, observed=Yobs)

    return pk_model
