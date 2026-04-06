from sklearn.base import BaseEstimator, TransformerMixin

from mlfs.base_algorithms.igmf import igmf

from sklearn.preprocessing import KBinsDiscretizer

import pandas as pd

import numpy as np


class PyIT_IGMF(BaseEstimator, TransformerMixin):

    def __init__(self, n_features):
        self.n_features = int(n_features)
        self.index_ranks = None
        
        
    def __str__(self):
        return 'PYIT_IGMF('+str(self.n_features)+')'
    
    
    def fit(self, dfX, dfy=None):  
        # discretizer
        kb_discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
        kb_discretizer.fit(dfX)
        # a saída é um dataframe por conta da configuração: set_config(transform_output="pandas") 
        new_dfX = kb_discretizer.transform(dfX)

        X = np.asarray(new_dfX, dtype=int)
        y = np.asarray(dfy, dtype=int)

        self.index_ranks = igmf().rank(X, y)
        
        return self
    

    def transform(self, dfX, dfy = None):
        indexes = self.index_ranks[:self.n_features]
        return dfX.iloc[:, indexes]