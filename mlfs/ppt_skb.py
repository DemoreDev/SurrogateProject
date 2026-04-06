import numpy as np

import pandas as pd

from mlfs.base_algorithms.FSAlgorithmsBasedLP import FSAlgorithmsBasedLP
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif

from sklearn.preprocessing import MinMaxScaler


class PPT_SelectKBest(FSAlgorithmsBasedLP, BaseEstimator, TransformerMixin):
    
    
    def __init__(self, n_features, method=f_classif):
        super().__init__()
        self.n_features = n_features
        self.method = method
        self.index_ranks = None
        
        
    def __str__(self):
        return 'PPT_SelectKBest('+str(self.n_features)+', '+self.method.__name__+')'
    
    
    def fit(self, dfX, dfy=None): 
        if self.method is chi2:
            scaler = MinMaxScaler().set_output(transform='pandas')
            dfX = scaler.fit_transform(dfX)
            
        # remove linhas cujos labels tem número de exemplos inferior ao threshould
        X_new, y_new = super().transformation_PPT(dfX.values, dfy.values)
        
        dfX_new = pd.DataFrame(X_new, columns=dfX.columns)
        
        # elimina colunas com mesmo valor
        vt = VarianceThreshold()
        vt.fit_transform(dfX_new)
        dfX_new = dfX_new[dfX_new.columns[vt.get_support(indices=True)]]

        # obtém X e Y (podem estar reduzidos)
        X = dfX_new.values
        y = y_new
        
        selector = SelectKBest(self.method, k=X.shape[1])
        selector.fit_transform(X, y)
        
        rank_features = np.argsort(selector.scores_)
        rank_features = rank_features[::-1] # ordena do maior score para o menor
        rank_features_cols = dfX_new.columns[rank_features]

        # encontra index de features no dataframe original
        self.index_ranks = [np.where(dfX.columns==col)[0][0] for col in rank_features_cols]
        
        return self
    

    def transform(self, dfX, dfy = None):
        indexes = self.index_ranks[:self.n_features]
        #print(indexes)
        return dfX.iloc[:, indexes]