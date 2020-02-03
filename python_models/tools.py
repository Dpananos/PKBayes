import pandas as pd
import numpy as np

def make_new_subjects(n_subjects=100):
    
    new_subjects_df = pd.DataFrame({'subjectids': np.arange(n_subjects), 'key':1})
    times_df = pd.DataFrame({'times': np.arange(0.5, 12, 1), 'key':1})
    test_times_df = pd.DataFrame({'times': np.arange(1, 12, 1), 'key':1})
    
    condition = (
        pd.merge(new_subjects_df, times_df, how='outer').
        sort_values('subjectids').
        drop('key', axis='columns').
        sort_values(['subjectids','times'])
    )
    
    no_condition = (
        pd.merge(new_subjects_df, test_times_df, how='outer').
        sort_values('subjectids').
        drop('key', axis='columns').
        sort_values(['subjectids','times'])
    )

    return condition, no_condition
    
    