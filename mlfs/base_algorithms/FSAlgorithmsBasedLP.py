import pandas as pd

import numpy as np


class FSAlgorithmsBasedLP():
    
    def __init__(self, threshold=6):
        self.threshold = threshold
            
    
    def transformation_PPT(self, X, y):  
        array_labels = [[line.__str__()[1:][:-1].replace(' ', '')] for line in y]
        y = np.append(y, array_labels, axis=1)
    
        unique, counts = np.unique(y[:, -1], return_counts=True)
        d = dict(zip(unique, counts))
    
        for key, count in d.items():
            if count < self.threshold:
                array_index = np.where(y[:, -1] == key)[0]
                X = np.delete(X, array_index, axis=0)
                y = np.delete(y, array_index, axis=0)
    
        l = 0
        labels = np.unique(y[:, -1])
        y_new = np.zeros_like(y[:, -1], dtype=int)
        for label in labels:
            for i in range(y.shape[0]):
                if y[i, -1] == label:
                    y_new[i] = l
            l += 1
    
        return X, y_new