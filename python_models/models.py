import numpy as np
import pymc3 as pm
from pymc3 import Model
import theano.tensor as tt


def pkfunc(CL, ke, ka, t):
    y = 2.5 * ke * ka / (CL * (ke - ka)) * (pm.math.exp(-ka * t) - pm.math.exp(-ke * t))
    return y


class WeakModel(Model):
    def __init__(self, yobs, times, subjectids, use_delay=True, name="", model=None):
        super().__init__(name, model)

        n = np.unique(subjectids).size
        if yobs is None:
            yobs = np.zeros_like(times)

        mu_CL = pm.Gamma("mu_CL", 55 / 2, 50 / 2)
        s_CL = pm.Gamma("s_CL", 15, 100)
        z_CL = pm.Normal("z_CL", 0, 1, shape=n)
        CL = pm.Deterministic("CL", pm.math.exp(mu_CL + z_CL * s_CL))

        mu_t = pm.Bound(pm.Normal, lower=0)("mu_t", pm.math.log(3), 0.15)
        s_t = pm.Gamma("s_t", 15, 100)
        z_t = pm.Normal("z_t", 0, 1, shape=n)
        tmax = pm.Deterministic("tmax", pm.math.exp(mu_t + z_t * s_t))

        alpha = pm.Beta("alpha", 2, 2, shape=n)
        ka = pm.Deterministic("ka", pm.math.log(alpha) / (tmax * (alpha - 1)))
        ke = pm.Deterministic("ke", alpha * ka)

        t = times
        if use_delay:
            phi = pm.Beta("phi", 2, 2)
            kappa = pm.Beta("kappa", 5, 5)
            delay = pm.Beta("delay", phi / kappa, (1 - phi) / kappa, shape=n)
            t = times - 0.5 * delay[subjectids]

        profile = pkfunc(CL[subjectids], ka[subjectids], ke[subjectids], t)
        ypred = pm.Deterministic("ypred", profile)
        sigma = pm.Lognormal("sigma", pm.math.log(0.1), 0.2)

        y = pm.Lognormal("y", pm.math.log(ypred), sigma, observed=yobs)

class WeakRegressionModel(Model):
    def __init__(self, yobs, times, X, subjectids, use_delay=True, name="", model=None):
        super().__init__(name, model)

        n,p = X.shape
        
        if yobs is None:
            yobs = np.zeros_like(times)

        mu_CL = pm.Gamma("mu_CL", 55 / 2, 50 / 2)
        betas_CL = pm.Normal('betas_CL', 0, 0.1, shape=p)
        s_CL = pm.Gamma("s_CL", 15, 100)
        z_CL = pm.Normal("z_CL", 0, 1, shape=n)
        CL = pm.Deterministic("CL", pm.math.exp(mu_CL + pm.math.dot(X, betas_CL) + z_CL * s_CL))

        mu_t = pm.Bound(pm.Normal, lower=0)("mu_t", pm.math.log(3), 0.15)
        betas_t = pm.Normal('betas_t',0, 0.1, shape=p)
        s_t = pm.Gamma("s_t", 15, 100)
        z_t = pm.Normal("z_t", 0, 1, shape=n)
        tmax = pm.Deterministic("tmax", pm.math.exp(mu_t + pm.math.dot(X, betas_t) + z_t * s_t))

        alpha = pm.Beta("alpha", 2, 2, shape=n)
        ka = pm.Deterministic("ka", pm.math.log(alpha) / (tmax * (alpha - 1)))
        ke = pm.Deterministic("ke", alpha * ka)

        t = times
        if use_delay:
            phi = pm.Beta("phi", 2, 2)
            kappa = pm.Beta("kappa", 5, 5)
            delay = pm.Beta("delay", phi / kappa, (1 - phi) / kappa, shape=n)
            t = times - 0.5 * delay[subjectids]

        profile = pkfunc(CL[subjectids], ka[subjectids], ke[subjectids], t)
        ypred = pm.Deterministic("ypred", profile)
        sigma = pm.Lognormal("sigma", pm.math.log(0.1), 0.2)

        y = pm.Lognormal("y", pm.math.log(ypred), sigma, observed=yobs)

class StrongModel(Model):
    def __init__(self, yobs, times, subjectids, test_times, test_subjectids, use_delay=True, name="", model=None):

        super().__init__(name, model)

        n = np.unique(subjectids).size

        if yobs is None:
            yobs = np.zeros_like(times)

        mu_CL = pm.Lognormal("mu_CL", 0.26, 0.04)
        s_CL = pm.Lognormal("s_CL", -1.14, 0.1)
        z_CL = pm.Normal("z_CL", 0, 1, shape=n)
        CL = pm.Deterministic("CL", pm.math.exp(mu_CL + z_CL * s_CL))

        mu_t = pm.Bound(pm.Normal, lower=0)("mu_t", 1, 0.05)
        s_t = pm.Lognormal("s_t", -1.34, 0.12)
        z_t = pm.Normal("z_t", 0, 1, shape=n)
        tmax = pm.Deterministic("tmax", pm.math.exp(mu_t + z_t * s_t))

        alpha = pm.Beta("alpha", 2, 2, shape=n)
        ka = pm.Deterministic("ka", pm.math.log(alpha) / (tmax * (alpha - 1)))
        ke = pm.Deterministic("ke", alpha * ka)

        t = times
        test_t = test_times
        if use_delay:
            phi = pm.Beta("phi", 41.6, 38.22)
            kappa = pm.Beta("kappa", 10.24, 9.1)
            delay = pm.Beta("delay", phi / kappa, (1 - phi) / kappa, shape=n)
            t = times - 0.5 * delay[subjectids]
            test_t = test_times - 0.5 * delay[test_subjectids]

        profile = pkfunc(CL[subjectids], ka[subjectids], ke[subjectids], t)
        ypred = pm.Deterministic("ypred", profile)

        predicted_profile = pkfunc(CL[test_subjectids], ka[test_subjectids], ke[test_subjectids], test_t)
        y_oos = pm.Deterministic("y_oos", predicted_profile)
        sigma = pm.Lognormal("sigma", -1.77, 0.06)

        y = pm.Lognormal("y", pm.math.log(ypred), sigma, observed=yobs)
