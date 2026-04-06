import numpy as np

import pandas as pd

from mlfs.base_algorithms.FSAlgorithmsBasedLP import FSAlgorithmsBasedLP
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.feature_selection import VarianceThreshold

from skrebate import ReliefF


class PPT_ReliefF(FSAlgorithmsBasedLP, BaseEstimator, TransformerMixin):
    
    
    def __init__(self, n_features, neighbors=10):
        super().__init__()
        self.n_features = n_features
        self.neighbors = neighbors # default weka
        self.index_ranks = None
    
    
    def __str__(self):
        return 'PPT_ReliefF('+str(self.n_features)+', '+str(self.neighbors)+')'
    
    
    def fit(self, dfX, dfy=None):    
        # remove linhas cujos labels tem número de exemplos inferior ao threshould
        X_new, y_new = super().transformation_PPT(dfX.values, dfy.values)
        dfX_new = pd.DataFrame(X_new, columns=dfX.columns)
        
        # elimina colunas com mesmo valor
        vt = VarianceThreshold()
        vt.fit_transform(dfX_new)
        dfX_new = dfX_new[dfX_new.columns[vt.get_support(indices=True)]]

        # obtém X e Y (podem estar reduzidos)
        X = np.asarray(dfX_new, dtype=float)
        y = y_new # já alterado por transformation_PPT
        
        fs = ReliefF(n_neighbors=self.neighbors, n_features_to_select=X.shape[1])
        fs.fit_transform(X, y)
        
        rank_features = np.argsort(fs.feature_importances_)
        rank_features = rank_features[::-1] # ordena do maior score para o menor
        rank_features_cols = dfX_new.columns[rank_features]

        # encontra index de features no dataframe original
        self.index_ranks = [np.where(dfX.columns==col)[0][0] for col in rank_features_cols]
        
        return self
    

    def transform(self, dfX, dfy = None):
        indexes = self.index_ranks[:self.n_features]
        #print(indexes)
        return dfX.iloc[:, indexes]