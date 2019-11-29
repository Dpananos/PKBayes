import pystan
import pickle
from hashlib import md5

def StanModel_cache(file, model_name=None, **kwargs):
    """Use just as you would `stan`"""
    with open(file, 'r') as f:
        model_code = f.read()
    code_hash = md5(model_code.encode('ascii')).hexdigest()
    if model_name is None:
        cache_fn = 'stan/cached-model-{}.pkl'.format(code_hash)
    else:
        cache_fn = 'stan/cached-{}-{}.pkl'.format(model_name, code_hash)
    try:
        sm = pickle.load(open(cache_fn, 'rb'))
    except:
        sm = pystan.StanModel(file=file)
        with open(cache_fn, 'wb') as f:
            pickle.dump(sm, f)
    else:
        print("Using cached StanModel")
    return sm
