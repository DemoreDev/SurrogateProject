import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.feature_selection import VarianceThreshold

from skrebate import ReliefF


class BR_ReliefF(BaseEstimator, TransformerMixin):    
    
    def __init__(self, n_features, neighbors=10):
        self.n_features = n_features
        self.neighbors = neighbors # default weka
        self.index_ranks = None
        
        
    def __str__(self):
        return 'BR_ReliefF('+str(self.n_features)+', '+str(self.neighbors)+')'
    
    
    def fit(self, dfX, dfy=None):    
        # elimina colunas com mesmo valor
        vt = VarianceThreshold()
        vt.fit_transform(dfX)
        dfX_new = dfX[dfX.columns[vt.get_support(indices=True)]]

        # obtém X (pode estar reduzido) e Y 
        X = np.asarray(dfX_new, dtype=float)
        y = np.asarray(dfy, dtype=int)
    
        values = []
        rlf = ReliefF(n_neighbors=self.neighbors, n_features_to_select=X.shape[1])

        for i in range(y.shape[1]):
            rlf.fit_transform(X, y[:, i])
            values.append(rlf.feature_importances_)

        mean_values = np.mean(values, axis=0)
        rank_features = np.argsort(mean_values)
        rank_features = rank_features[::-1]
        rank_features_cols = dfX_new.columns[rank_features]
        
        # encontra index de features no dataframe original
        self.index_ranks = [np.where(dfX.columns==col)[0][0] for col in rank_features_cols]
        
        return self
    

    def transform(self, dfX, dfy = None):
        indexes = self.index_ranks[:self.n_features]
        #print(indexes)
        return dfX.iloc[:, indexes]