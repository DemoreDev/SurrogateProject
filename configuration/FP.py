# FP - Feature Preprocessing

import numpy as np

#from sklearn.decomposition import PCA
#from sklearn.decomposition import KernelPCA
#from sklearn.decomposition import TruncatedSVD
#from sklearn.decomposition import FastICA

from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import PolynomialFeatures

from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from mlfs.br_skb import BR_SelectKBest
from mlfs.ppt_skb import PPT_SelectKBest
from mlfs.br_relieff import BR_ReliefF
from mlfs.ppt_relieff import PPT_ReliefF
from mlfs.ppt_sfm import PPT_SelectFromModel
from mlfs.ppt_rfe import PPT_RecursiveFeatureElimination

from mlfs.igmf_adapted import PyIT_IGMF
from mlfs.ppt_mi_adapted import PyIT_PPT_MI
from mlfs.scls_adapted import PyIT_SCLS
from mlfs.lrfs_adapted import PyIT_LRFS
from mlfs.lsmfs_adapted import PyIT_LSMFS
from mlfs.mlsmfs_adapted import PyIT_MLSMFS
from mlfs.d2f_adapted import PyIT_D2F
from mlfs.pmu_adapted import PyIT_PMU
from mlfs.mdmr_adapted import PyIT_MDMR

from sklearn.feature_selection import f_classif, chi2, mutual_info_classif


def get_config(n_features):

    FS_config_dic = {
         
        'NO_FEATURE_PREPROCESSING': {},

        # FS - PyIT library
        'mlfs.igmf_adapted.PyIT_IGMF': {
            'n_features': get_array_features(n_features) 
        },

        'mlfs.ppt_mi_adapted.PyIT_PPT_MI': {
            'n_features': get_array_features(n_features), 
            #'prune_threshold': np.array([4, 5, 6])
        },
        
        'mlfs.scls_adapted.PyIT_SCLS': {
            'n_features': get_array_features(n_features)
        },

        'mlfs.lrfs_adapted.PyIT_LRFS': {
            'n_features': get_array_features(n_features)
        },

        'mlfs.lsmfs_adapted.PyIT_LSMFS': {
            'n_features': get_array_features(n_features)
        },

        'mlfs.mlsmfs_adapted.PyIT_MLSMFS': {
            'n_features': get_array_features(n_features)
        },

        'mlfs.d2f_adapted.PyIT_D2F': {
            'n_features': get_array_features(n_features)
        },

        'mlfs.pmu_adapted.PyIT_PMU': {
            'n_features': get_array_features(n_features)
        },

        'mlfs.mdmr_adapted.PyIT_MDMR': {
            'n_features': get_array_features(n_features)
        },

        # FS - SCIKIT

        'sklearn.feature_selection.SelectFromModel': {
           'estimator': np.array(['sklearn.ensemble.ExtraTreesClassifier()', 'sklearn.ensemble.RandomForestClassifier()']), 
            'threshold' : np.array(['\'mean\'']),
            #'prefit' : np.array([True])
        },

        'sklearn.feature_selection.RFE': {
            'estimator': np.array(['sklearn.ensemble.ExtraTreesClassifier()', 'sklearn.ensemble.RandomForestClassifier()']), 
            'n_features_to_select': get_array_features(n_features)
        },

        # FS 
        
        'mlfs.br_skb.BR_SelectKBest': {
            'n_features': get_array_features(n_features), 
            'method': np.array(['sklearn.feature_selection.f_classif', 'sklearn.feature_selection.chi2', 'sklearn.feature_selection.mutual_info_classif'])
        },

        
        'mlfs.ppt_skb.PPT_SelectKBest': {
            'n_features': get_array_features(n_features), 
            'method': np.array(['sklearn.feature_selection.f_classif', 'sklearn.feature_selection.chi2', 'sklearn.feature_selection.mutual_info_classif'])
        },

        'mlfs.br_relieff.BR_ReliefF': {
            'n_features': get_array_features(n_features),
            'neighbors': np.array([10]), # VER
        },

        'mlfs.ppt_relieff.PPT_ReliefF': {
            'n_features': get_array_features(n_features),
            'neighbors': np.array([10]),  
        },

        'mlfs.ppt_sfm.PPT_SelectFromModel': {
            'n_features': get_array_features(n_features),
            'estimator': np.array(['sklearn.ensemble.ExtraTreesClassifier()', 'sklearn.ensemble.RandomForestClassifier()']), 
        },

        'mlfs.ppt_rfe.PPT_RecursiveFeatureElimination': {
            'n_features': get_array_features(n_features),
            'estimator': np.array(['sklearn.ensemble.ExtraTreesClassifier()', 'sklearn.ensemble.RandomForestClassifier()']), 
        },
         
        # OTHERS
        

    }

    return FS_config_dic


def get_array_features(n_features):
        inf = sup = 0
        
        if n_features<100:
            inf = 0.4
            sup = 0.8
        elif n_features>=100 and n_features<500:
            inf = 0.3
            sup = 0.6
        elif n_features>=500 and n_features<1000:
            inf = 0.2
            sup = 0.4
        elif n_features >= 1000:
            inf = 0.1
            sup = 0.2
            
        inf = int(n_features*inf)
        sup = int(n_features*sup)
        
        return np.arange(inf, sup+1, dtype=int)


def get_algorithms(n_features):
    return np.array(list(get_config(n_features).keys()))
