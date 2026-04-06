import numpy as np

import pandas as pd

from mlfs.base_algorithms.FSAlgorithmsBasedLP import FSAlgorithmsBasedLP
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE

from sklearn.ensemble import RandomForestClassifier


class PPT_RecursiveFeatureElimination(FSAlgorithmsBasedLP, BaseEstimator, TransformerMixin):
    
    
    def __init__(self, n_features, estimator=RandomForestClassifier()):
        super().__init__()
        self.n_features = n_features
        self.estimator = estimator
        self.index_ranks = None
        
    
    def __str__(self):
        return 'PPT_RecursiveFeatureElimination('+str(self.n_features)+', '+str(self.estimator)+')'
    
    
    def fit(self, dfX, dfy=None): 
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
      
        # um classificador é treinado no conjunto inicial de recursos e a importância de cada recurso 
        # é obtida por meio de qualquer atributo específico (como coef_, feature_importances_).
        selector = RFE(self.estimator, n_features_to_select=1, step=1)
        selector.fit_transform(X, y)
      
        rank_features = selector.ranking_ # ranqueia features
        rank_features = np.argsort(rank_features) # ordena feature por index
        rank_features_cols = dfX_new.columns[rank_features] # obtém nome da feature no dataframe novo (reduzido)
        
        # encontra index de features no dataframe original
        self.index_ranks = [np.where(dfX.columns==col)[0][0] for col in rank_features_cols]
          
        return self
    

    def transform(self, dfX, dfy = None):
        indexes = self.index_ranks[:self.n_features]
        #print(indexes)
        return dfX.iloc[:, indexes]