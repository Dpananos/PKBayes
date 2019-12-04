import numpy as np
import pandas as pd
from scipy.stats import norm
import pymc3 as pm

np.random.seed(19920908)

def generate_covariates(n_subjects, n_continuous_predictors, n_bianry_predictors, binary_prob):

    corr = pm.LKJCorr.dist(eta=1, n=n_continuous_predictors).random()

    Sigma = np.eye(n_continuous_predictors)

    Sigma[np.tril_indices(n_continuous_predictors,k=-1)]=corr
    Sigma[np.triu_indices(n_continuous_predictors,k=1)]=corr


    X_continuous = pm.MvNormal.dist(mu=np.zeros(n_continuous_predictors), cov=Sigma).random(size = n_subjects)
    X_binary = pm.Bernoulli.dist(p=binary_prob, shape=(n_subjects, n_bianry_predictors)).random()
    X = np.c_[X_continuous, X_binary]

    return X


def generate_regression_coefficients(X, beta_cov):
    n,p = X.shape
    betas = pm.MvNormal.dist(mu=np.zeros(3), cov=np.diag(beta_cov), shape=p).random(size = p)

    return betas

def generate_pk_params(X,beta, beta_baseline, use_normal, rfx_cov):

    baseline = np.array(beta_baseline)
    effects = X@beta

    if use_normal:
        Z = pm.Normal.dist().random(size =effects.shape)
    else:
        Z = pm.StudentT.dist(nu=5).random(size = effects.shape)

    pk_params = np.exp(baseline + effects + Z@np.diag(rfx_cov) )

    return pk_params


def pk_func(t,D,Cl, ka, ke):

    return (D/Cl)*(ke*ka/(ke-ka))*(np.exp(-ka*t) - np.exp(-ke*t))


def make_obs(D, subjects, times, X, pk, sigma_obs):

    unique_subjects = np.sort(np.unique(subjects))
    pkdf = pd.DataFrame(pk, columns=['Cl','ka','ke'], index=unique_subjects )
    tdf = pd.DataFrame(times, columns=['t'], index=subjects)
    Xdf = pd.DataFrame(X, columns=[f'x_{j}' for j in range(X.shape[1])], index=unique_subjects )

    df = tdf.join(pkdf, how='left').join(Xdf, how='left')
    df['y'] = pk_func(df.t, D, df.Cl, df.ka, df.ke )
    z = norm(loc=0, scale=sigma_obs)
    df['y_obs'] = np.exp( np.log(df.y) + z.rvs(size=df.shape[0], random_state= 0) )

    return df