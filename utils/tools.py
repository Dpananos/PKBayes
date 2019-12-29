import numpy as np 
import pickle


def generate_times_subjects(t, subjects, random_sample = False):
    #TODO: Mix and match version.
    '''Function to return times and subject Ids for convienience.  
    
    Inputs:
    t -- array of desired observation times or int for number of observation times between 0.5 and 12
    subjects -- integer for number of subjects to create
    
    '''

    if isinstance(t, int) and not random_sample:
        times = np.tile(np.linspace(0.5,12, t), subjects)
        subject_ids = np.concatenate([t*[j] for j in range(subjects)])
    elif isinstance(t, int) and random_sample:
        times = np.concatenate([np.sort(np.random.uniform(low=0, high=12, size=t)) for j in range(subjects)])
        subject_ids = np.concatenate([t*[j] for j in range(subjects)])
    else:
        times = np.tile(t, subjects)
        subject_ids = np.concatenate([len(t)*[j] for j in range(subjects)])

    return times, subject_ids


def save_obj(obj, name):
    with open(f'{name}.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(f'{name}.pkl', 'rb') as f:
        return pickle.load(f)


def summarize_posterior(data, var_name, model_name):
    
    pred = (
            data.
            posterior[var_name].
            to_dataframe().
            groupby(level=2).  #This should be the dimension
            agg([('pred',np.mean), 
                ('low', lambda x: np.quantile(x,0.025)),
                ('high', lambda x: np.quantile(x,0.975)) ])
        )

    pred.columns = pred.columns.droplevel()
    pred.index.name = None
    pred.columns = [f'{model_name}_{j}' for j in pred.columns]

    return pred

def summarize_ppc(data, var_name, model_name):
    
    pred = (
            data.
            posterior_predictive[var_name].
            to_dataframe().
            groupby(level=2).  #This should be the dimension
            agg([('pred',np.mean), 
                ('low', lambda x: np.quantile(x,0.025)),
                ('high', lambda x: np.quantile(x,0.975)) ])
        )

    pred.columns = pred.columns.droplevel()
    pred.index.name = None
    pred.columns = [f'{model_name}_{j}' for j in pred.columns]

    return pred


