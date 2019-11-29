import numpy as np
import pandas as pd

def pk_func(t,D,Cl, ka, ke):

    return (D/Cl)*(ke*ka/(ke-ka))*(np.exp(-ka*t) - np.exp(-ke*t))


def make_obs(D, subjects, times, X, pk, sigma_obs):

    unique_subjects = np.sort(np.unique(subjects))
    pkdf = pd.DataFrame(pk, columns = ['Cl','ka','ke'], index = unique_subjects )
    tdf = pd.DataFrame(times, columns = ['t'], index = subjects)
    Xdf =  pd.DataFrame(X, columns = [f'x_{j}' for j in range(X.shape[1])], index = unique_subjects )

    df = tdf.join(pkdf, how = 'left').join(Xdf, how='left')
    df['y'] = pk_func(df.t, D, df.Cl, df.ka, df.ke )
    df['y_obs'] = np.exp( np.log(df.y) + np.random.normal(0, sigma_obs, size = df.shape[0]) )

    return df