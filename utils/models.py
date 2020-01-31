import numpy as np
import pymc3 as pm
import theano.tensor as tt

def model_factory(Yobs, times, subject_ids):
    # This is the case when I am simulating from the prior predictive.
    if Yobs is None:
        Yobs = np.zeros_like(times)

    with pm.Model() as pk_model:

        #Parameter for the intercept
        mu_CL = pm.Gamma('mu_CL',12, 10)
        sigma_CL = pm.Gamma('sigma_CL',10, 10)
        z_CL = pm.Normal('z_CL', 0, 1, shape = np.unique(subject_ids).size)
        CL = pm.Deterministic("CL", pm.math.exp(mu_CL + z_CL*sigma_CL))


        L = np.eye(2)
        z = pm.Normal('z', 0, 1, shape=2)
        MU = np.array([-2,0]) + pm.math.dot(L,z)

        mu_ke = pm.Deterministic('mu_ke',MU[0])
        sigma_ke= pm.Gamma('sigma_ke',10, 10)
        z_ke = pm.Normal('z_ke', 0, 1, shape = np.unique(subject_ids).size)
        ke = pm.Deterministic("ke", pm.math.exp(mu_ke + z_ke*sigma_ke))

        mu_ka = pm.Deterministic('mu_ka',MU[1])
        sigma_ka= pm.Gamma('sigma_ka',10, 10)
        z_ka = pm.Normal('z_ka', 0, 1, shape = np.unique(subject_ids).size)
        ka = pm.Deterministic("ka", pm.math.exp(mu_ka + z_ka*sigma_ka))


        phi = pm.Uniform('delays_mu', 0, 1)
        kappa = pm.HalfCauchy('delays_kappa',1)
        delays = pm.Beta('delays',phi*kappa, (1-phi)*kappa, shape = np.unique(subject_ids).size)

        t = times - 0.5*delays[subject_ids]

        y_est = (
            2.5
            / CL[subject_ids]
            * (ke[subject_ids] * ka[subject_ids])
            / (ke[subject_ids] - ka[subject_ids])
            * (tt.exp(-ka[subject_ids] * t) - tt.exp(-ke[subject_ids] * t))
        )

        y_conc = pm.Deterministic("y_est", y_est)
        sigma = pm.Lognormal("sigma", tt.log(0.1) ,0.2)

        y = pm.Lognormal("Yobs", tt.log(y_est), sigma, observed=Yobs)
        
    return pk_model
        


def strong_model_factory(Yobs, times, subject_ids, test_times, test_subject_ids):

    """Returns a model context for a strongly informative model.
    
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

    with pm.Model() as pk_model:
        #Parameter for the intercept
        mu_CL = pm.Gamma('mu_CL',1.295/0.073, 1.295**2/0.073)
        sigma_CL = pm.Lognormal('sigma_CL',-0.82,0.12)
        z_CL = pm.Normal('z_CL', 0, 1, shape = np.unique(subject_ids).size)
        CL = pm.Deterministic("CL", pm.math.exp(mu_CL + z_CL*sigma_CL))

        L = np.array([[ 0.05,  0.  ],[-0.03,  0.14]])
        z = pm.Normal('z', 0, 1, shape=2)
        MU = np.array([-1.83,0]) + pm.math.dot(L,z)

        mu_ke = pm.Deterministic('mu_ke',MU[0])
        sigma_ke= pm.Lognormal('sigma_ke',-1.47, 0.21)
        z_ke = pm.Normal('z_ke', 0, 1, shape = np.unique(subject_ids).size)
        ke = pm.Deterministic("ke", pm.math.exp(mu_ke + z_ke*sigma_ke))

        mu_ka = pm.Deterministic('mu_ka',MU[1])
        sigma_ka= pm.Lognormal('sigma_ka',-0.28, 0.142)
        z_ka = pm.Normal('z_ka', 0, 1, shape = np.unique(subject_ids).size)
        ka = pm.Deterministic("ka", pm.math.exp(mu_ka + z_ka*sigma_ka))


        phi = pm.Beta('delays_mu',  59.829732, 34.310231)
        kappa = pm.Lognormal('delays_kappa',1.07, 0.2)
        delays = pm.Beta('delays',phi*kappa, (1-phi)*kappa, shape = np.unique(subject_ids).size)

        t = times - 0.5*delays[subject_ids]
        delayed_test_times = test_times - 0.5*delays[test_subject_ids]

        y_est = (
            2.5
            / CL[subject_ids]
            * (ke[subject_ids] * ka[subject_ids])
            / (ke[subject_ids] - ka[subject_ids])
            * (tt.exp(-ka[subject_ids] * t) - tt.exp(-ke[subject_ids] * t))
        )
        
        
        y_oos_pred = (
                2.5
                / CL[test_subject_ids]
                * (ke[test_subject_ids] * ka[test_subject_ids])
                / (ke[test_subject_ids] - ka[test_subject_ids])
                * (tt.exp(-ka[test_subject_ids] * delayed_test_times) - tt.exp(-ke[test_subject_ids] * delayed_test_times))
            )

        y_conc = pm.Deterministic("y_est", y_est)
        y_pred = pm.Deterministic('y_pred', y_oos_pred)
        sigma = pm.Lognormal("sigma", -2 ,0.05)

        y = pm.Lognormal("Yobs", tt.log(y_est), sigma, observed=Yobs)
        return pk_model

    

def estimated_from_tmax(Yobs, times, subject_ids):
    # This is the case when I am simulating from the prior predictive.
    if Yobs is None:
        Yobs = np.zeros_like(times)

    with pm.Model() as pk_model:

        #Parameter for the intercept
        mu_CL = pm.Gamma('mu_CL',45, 50)
        sigma_CL = pm.Gamma('sigma_CL',15, 100)
        z_CL = pm.Normal('z_CL', 0, 1, shape = np.unique(subject_ids).size)
        CL = pm.Deterministic("CL", pm.math.exp(mu_CL + z_CL*sigma_CL))
        
        mu_t = pm.Lognormal('mu_t', pm.math.log(3.3), 0.25)
        s_t = pm.Gamma('s_t', 15,100)

        z = pm.Normal('z', 0, 1, shape = np.unique(subject_ids).size )
        t = pm.math.exp(pm.math.log(3.3) + z*0.15)
        alpha= pm.Beta('alpha', 2,2, shape = np.unique(subject_ids).size)

        
        ka = pm.math.log(alpha)/(t*(alpha-1))
        ke = alpha*ka


#         phi = pm.Beta('delays_mu', 1, 1)
#         kappa = pm.Beta('delays_kappa',2, 2)
#         delays = pm.Beta('delays',phi/kappa, (1-phi)/kappa, shape = np.unique(subject_ids).size)

        t = times 

        y_est = (
            2.5
            / CL[subject_ids]
            * (ke[subject_ids] * ka[subject_ids])
            / (ke[subject_ids] - ka[subject_ids])
            * (tt.exp(-ka[subject_ids] * t) - tt.exp(-ke[subject_ids] * t))
        )

        y_conc = pm.Deterministic("y_est", y_est)
        sigma = pm.Lognormal("sigma", tt.log(0.1) ,0.2)

        y = pm.Lognormal("Yobs", tt.log(y_est), sigma, observed=Yobs)
        
    return pk_model

def strongly_estimated_from_tmax(Yobs, times, subject_ids, test_times, test_subject_ids):
    # This is the case when I am simulating from the prior predictive.
    if Yobs is None:
        Yobs = np.zeros_like(times)

    with pm.Model() as pk_model:

        #Parameter for the intercept
        mu_CL = pm.Lognormal('mu_CL',0.24, 0.042)
        sigma_CL = pm.Lognormal('sigma_CL',-1.15, 0.1)
        z_CL = pm.Normal('z_CL', 0, 1, shape = np.unique(subject_ids).size)
        CL = pm.Deterministic("CL", pm.math.exp(mu_CL + z_CL*sigma_CL))
        

        z = pm.Normal('z', 0, 1, shape = np.unique(subject_ids).size )
        tmax = pm.Deterministic('tmax', pm.math.exp(pm.math.log(3.3) + z*0.15))
        alpha= pm.Beta('alpha', 2,2, shape = np.unique(subject_ids).size)
        
        ka = pm.Deterministic('ka',pm.math.log(alpha)/(tmax*(alpha-1)))
        ke = pm.Deterministic('ke',alpha*ka)


#         phi = pm.Beta('delays_mu', 31.5, 40.48)
#         kappa = pm.Beta('delays_kappa',7.18, 3.72)
#         delays = pm.Beta('delays',phi/kappa, (1-phi)/kappa, shape = np.unique(subject_ids).size)

#         t = times - 0.5*delays[subject_ids]
        t = times
#         delayed_test_times = test_times -0.5*delays[test_subject_ids]
        delayed_test_times = test_times

        y_est = (
            2.5
            / CL[subject_ids]
            * (ke[subject_ids] * ka[subject_ids])
            / (ke[subject_ids] - ka[subject_ids])
            * (tt.exp(-ka[subject_ids] * t) - tt.exp(-ke[subject_ids] * t))
        )
        
        
        cmax = (
            2.5
            / CL
            * (ke * ka)
            / (ke - ka)
            * (tt.exp(-ka * tmax) - tt.exp(-ke * tmax))
        )
        
        y_oos_pred = (
                2.5
                / CL[test_subject_ids]
                * (ke[test_subject_ids] * ka[test_subject_ids])
                / (ke[test_subject_ids] - ka[test_subject_ids])
                * (tt.exp(-ka[test_subject_ids] * delayed_test_times) - tt.exp(-ke[test_subject_ids] * delayed_test_times))
            )

        y_conc = pm.Deterministic("y_est", y_est)
        y_pred = pm.Deterministic("y_pred", y_oos_pred)
        Cmax = pm.Deterministic('Cmax', cmax)
        sigma = pm.Lognormal("sigma", -1.57 ,0.06)

        y = pm.Lognormal("Yobs", tt.log(y_est), sigma, observed=Yobs)
        
    return pk_model