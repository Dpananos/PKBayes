import pymc3 as pm
import pandas as pd  
import numpy as np
import arviz as az
import theano.tensor as tt

def pk_regression(df,**kwargs):

    df = pd.read_csv('data/test.csv', index_col = 0)
    df.index = df.index.rename('subject')
    df = df.reset_index()
    df['subject'] = df.subject.astype('category').cat.codes

    idx = df.subject.values
    t = df.t.values
    yobs = df.y_obs.values
    X = df.drop_duplicates('subject').loc[:, df.columns[df.columns.str.contains('x')]]

    n,p = X.shape

    with pm.Model() as pkmodel:
        
        baseline_cl = pm.Bound(pm.Normal, lower=0)('baseline_cl', mu = tt.log(3), sd=1)
        baseline_ke = pm.Normal('baseline_ke', mu = 0, sd=1)
        alpha = pm.Beta('alpha',20,20)
        baseline_ka = pm.Deterministic('baseline_ka', tt.log(1.0/alpha) + baseline_ke)
        # baseline_ka = pm.Bound(pm.Normal, lower = baseline_ke)('baseline_ka', mu = 0, sd = 1)
        
        baseline = tt.stack([baseline_cl, baseline_ka, baseline_ke])
        sigma = pm.Gamma('sigma',.5,1)
        S = pm.HalfCauchy('S',1, shape=3)
        z = pm.Normal('z', mu = 0, sd = 1, shape= (n,3))
        
        beta = pm.Normal('beta', mu=0, sd=0.1, shape = (p,3))
        params = pm.Deterministic('params', tt.exp(baseline + tt.dot(X,beta) + tt.dot(z, tt.diag(S))))

    #     params = pm.Deterministic('params', tt.exp(baseline + tt.dot(z, tt.diag(S))))

        CL = params[tuple(idx), 0]
        KA = params[tuple(idx), 1]
        KE = params[tuple(idx), 2]
        
        delay_mu = pm.Beta('mu', 1,1)
        delay_kappa = pm.Pareto('kappa', m=10, alpha=1)
        delay = pm.Beta('delay', alpha = delay_mu*delay_kappa, beta=(1-delay_mu)*delay_kappa, shape=n)
        dt = t-0.5*delay[idx]
        
        c = (5/CL)*(KE*KA)/(KE-KA)*(tt.exp(-KA*dt) - tt.exp(-KE*dt))
        concentration = pm.Deterministic('conc', c)
        
        Y = pm.Lognormal('Yobs', mu = tt.log(concentration), sd = sigma, observed = yobs)
        
        prior = pm.sample_prior_predictive(samples=1)
        trace = pm.sample(**kwargs)
        posterior = az.from_pymc3(trace)


    return posterior



def pk_mixed_model(df,**kwargs):
    
    df = pd.read_csv('data/test.csv', index_col = 0)
    df.index = df.index.rename('subject')
    df = df.reset_index()
    df['subject'] = df.subject.astype('category').cat.codes

    idx = df.subject.values
    t = df.t.values
    yobs = df.y_obs.values
    X = df.drop_duplicates('subject').loc[:, df.columns[df.columns.str.contains('x')]]

    n,p = X.shape

    with pm.Model() as pkmodel:
        
        baseline_cl = pm.Bound(pm.Normal, lower=0)('baseline_cl', mu = tt.log(3), sd=1)
        baseline_ke = pm.Normal('baseline_ke', mu = 0, sd=1)
        alpha = pm.Beta('alpha',20,20)
        baseline_ka = pm.Deterministic('baseline_ka', tt.log(1.0/alpha) + baseline_ke)

        # baseline_ka = pm.Bound(pm.Normal, lower = baseline_ke)('baseline_ka', mu = 0, sd=1)
        
        baseline = tt.stack([baseline_cl, baseline_ka, baseline_ke])
        sigma = pm.Gamma('sigma',.5,1)
        S = pm.HalfCauchy('S',1, shape=3)
        z = pm.Normal('z' , shape= (n,3))
        
        beta = pm.Normal('beta', mu=0, sd=0.1, shape = (p,3))
        # params = pm.Deterministic('params', tt.exp(baseline + tt.dot(X,beta) + tt.dot(z, tt.diag(S))))

        params = pm.Deterministic('params', tt.exp(baseline + tt.dot(z, tt.diag(S))))

        CL = params[tuple(idx), 0]
        KA = params[tuple(idx), 1]
        KE = params[tuple(idx), 2]
        
        delay_mu = pm.Beta('mu',1,1)
        delay_kappa = pm.Pareto('kappa', m=5, alpha=1)
        delay = pm.Beta('delay', alpha = delay_mu*delay_kappa, beta=(1-delay_mu)*delay_kappa, shape=n)
        dt = t-0.5*delay[idx]
        
        c = (5/CL)*(KE*KA)/(KE-KA)*(tt.exp(-KA*dt) - tt.exp(-KE*dt))
        concentration = pm.Deterministic('conc', c)
        
        Y = pm.Lognormal('Yobs', mu = tt.log(concentration), sd = sigma, observed = yobs)
        
        prior = pm.sample_prior_predictive(samples=1)
        trace = pm.sample(**kwargs)
        posterior = az.from_pymc3(trace)


    return posterior

