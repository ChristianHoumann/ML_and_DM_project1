import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd
from matplotlib.pyplot import (figure, plot, title, xlabel, ylabel, show,
                               legend, subplot, xticks, yticks, boxplot, hist,
                               ylim)

# read the data into python
df = pd.read_csv("http://www-stat.stanford.edu/~tibs/ElemStatLearn/datasets/SAheart.data",
	encoding='utf-8')

raw_data = df._get_values[:]


cols = range(1, 10)
X = raw_data[:, cols]
y = raw_data[:, 10]

famHistLable = raw_data[:,5] # -1 takes the last column

famHistNames = np.unique(famHistLable)

famHistDict = dict(zip(famHistNames,range(len(famHistNames))))

Xr = X.copy();
Xr[:,4] = np.array([famHistDict[cl] for cl in famHistLable])

attributeNames = np.asarray(df.columns[range(1, 11)])
## a short version of attributenames:
attributeNamesShort = [None] * 10
for i in range(len(attributeNames)):
    attributeNamesShort[i] = (attributeNames[i])[:3]
    
N, M = X.shape

# make array of all atributes
Xall = np.zeros((N,M+1))
Xall[:,0:M] = Xr
Xall[:,M] = y


#Make standadized values of Xr (subtract mean and devide by standard deviation)
Y = (Xr - np.ones((N,1))*Xr.mean(axis=0))
Y = np.array(Y,dtype=float)
Y = Y*(1/np.std(Y,0))

Yall = (Xall - np.ones((N,1))*Xall.mean(axis=0))
Yall = np.array(Yall,dtype=float)
Yall = Yall*(1/np.std(Yall,0))


############ Regression ##################










############ Classification ##################






















