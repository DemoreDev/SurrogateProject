import numpy as np

import configuration.MLC as config_mlc


def get_config(n_features, n_labels):
    
    MLC_ensemble_config_dic = {}   
    
    return MLC_ensemble_config_dic


def get_algorithms(n_features, n_labels):
    return np.array(list(get_config(n_features, n_labels).keys()))