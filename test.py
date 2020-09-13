# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 09:47:25 2020

@author: yomama
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend


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

N, M = X.shape

# Scatterplot of two attributes

i = 1
j = 7  
plt.title('tobacco against alcohol')
plt.plot(Xr[:, i], Xr[:, j], 'o')
plt.xlabel(attributeNames[i]);
plt.ylabel(attributeNames[j]);
plt.show()

i = 6
j = 8  
plt.title('are older people fat')
plt.plot(Xr[:, i], Xr[:, j], 'o')
plt.xlabel(attributeNames[i]);
plt.ylabel(attributeNames[j]);
plt.show()

i = 3
j = 6  
plt.title('correlation between adiposity and BMI')
plt.plot(Xr[:, i], Xr[:, j], 'o')
plt.xlabel(attributeNames[i]);
plt.ylabel(attributeNames[j]);
plt.show()

i = 2
j = 3  
plt.title('ldl and adiposity')
plt.plot(Xr[:, i], Xr[:, j], 'o')
plt.xlabel(attributeNames[i]);
plt.ylabel(attributeNames[j]);
plt.show()

i = 0
j = 3  
plt.title('sbp and adiposity')
plt.plot(Xr[:, i], Xr[:, j], 'o')
plt.xlabel(attributeNames[i]);
plt.ylabel(attributeNames[j]);
plt.show()

i = 5
j = 0  
plt.title('Type-a and sbp')
plt.plot(Xr[:, i], Xr[:, j], 'o')
plt.xlabel(attributeNames[i]);
plt.ylabel(attributeNames[j]);
plt.show()

i = 5
j = 1  
plt.title('does Type-a people smoke')
plt.plot(Xr[:, i], Xr[:, j], 'o')
plt.xlabel(attributeNames[i]);
plt.ylabel(attributeNames[j]);
plt.show()

i = 5
j = 7
plt.title('does Type-a people drink')
plt.plot(Xr[:, i], Xr[:, j], 'o')
plt.xlabel(attributeNames[i]);
plt.ylabel(attributeNames[j]);
plt.show()


### PCA analysis
Y = (Xr - np.ones((N,1))*Xr.mean(axis=0))
Y = np.array(Y,dtype=float)
Y = Y*(1/np.std(Y,0))
# TODO: DOES THIS MAKE SENSE FOR FAMHIST


# PCA by computing SVD of Y
U,S,Vh = svd(Y,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum()

threshold = 0.9

# how much variance does the first three components explain
(rho[0:3]).sum()

plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()


#### plot PC with the data
#### TODO: WHAD SKER DER!
V = Vh.T

Z = Y @ V

# Indices of the principal components to be plotted
i = 0
j = 1

f = figure()
title('Heart disease data: PCA')
#Z = array(Z)
for c in range(2):
    # select indices belonging to class c:
    class_mask = y==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
legend(["Negative","Positive"])
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

# Output result to screen
show()



# Plot attribute coefficients in principal component space
for att in range(V.shape[1]):
    plt.arrow(0,0, V[att,i], V[att,j])
    plt.text(V[att,i], V[att,j], attributeNames[att])
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.xlabel('PC'+str(i+1))
plt.ylabel('PC'+str(j+1))
plt.grid()
# Add a unit circle
plt.plot(np.cos(np.arange(0, 2*np.pi, 0.01)), 
    np.sin(np.arange(0, 2*np.pi, 0.01)));
plt.axis('equal')
plt.show()

i = 2
j = 3

for att in range(V.shape[1]):
    plt.arrow(0,0, V[att,i], V[att,j])
    plt.text(V[att,i], V[att,j], attributeNames[att])
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.xlabel('PC'+str(i+1))
plt.ylabel('PC'+str(j+1))
plt.grid()
# Add a unit circle
plt.plot(np.cos(np.arange(0, 2*np.pi, 0.01)), 
    np.sin(np.arange(0, 2*np.pi, 0.01)));
plt.axis('equal')
plt.show()
