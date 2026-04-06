import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif

from sklearn.preprocessing import MinMaxScaler


class BR_SelectKBest(BaseEstimator, TransformerMixin):
        
    def __init__(self, n_features, method=f_classif):
        self.n_features = n_features
        self.method = method
        self.index_ranks = None
        

    def __str__(self):
        return 'BR_SelectKBest('+str(self.n_features)+','+self.method.__name__+')'
    
    
    def fit(self, dfX, dfy=None): # train data   
        if self.method is chi2:
            scaler = MinMaxScaler().set_output(transform='pandas')
            dfX = scaler.fit_transform(dfX)
    
        # elimina colunas com mesmo valor
        vt = VarianceThreshold()
        vt.fit_transform(dfX)
        dfX_new = dfX[dfX.columns[vt.get_support(indices=True)]]
        
        # obtém X (pode estar reduzido) e Y 
        X = dfX_new.values
        y = dfy.values
    
        # calcula estatística das features com cada label
        statistic_values = np.array([])
        for i in range(y.shape[1]):
            if self.method is mutual_info_classif: 
                statistics = self.method(X, y[:, i]) # mi
            else:
                statistics, p_value = self.method(X, y[:, i])

            if len(statistic_values) == 0:
                statistic_values = statistics
            else:
                statistic_values = np.vstack((statistic_values, statistics))
        
        # encontra estatística média
        statistic_mean = np.mean(statistic_values, axis=0)
        
        # encontra features com maiores estatísticas
        rank_features = np.argsort(statistic_mean)
        rank_features = rank_features[::-1]
        rank_features_cols = dfX_new.columns[rank_features]

        # encontra index de features no dataframe original
        self.index_ranks = [np.where(dfX.columns==col)[0][0] for col in rank_features_cols]
        
        return self
    

    def transform(self, dfX, dfy = None):
        indexes = self.index_ranks[:self.n_features]
        #print(indexes)
        return dfX.iloc[:, indexes]