import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from toolbox_02450 import rocplot, confmatplot
from sklearn import model_selection
from toolbox_02450 import rlr_validate


# read the data into python
df = pd.read_csv("http://www-stat.stanford.edu/~tibs/ElemStatLearn/datasets/SAheart.data",
	encoding='utf-8')

raw_data = df._get_values[:]


cols = range(1, 10)
X = raw_data[:, cols]
y = raw_data[:, 10]
y = y.astype(np.float)

famHistLable = raw_data[:,5] # -1 takes the last column

famHistNames = np.unique(famHistLable)

famHistDict = dict(zip(famHistNames,range(len(famHistNames))))

Xr = X.copy();
Xr[:,4] = np.array([famHistDict[cl] for cl in famHistLable])

Xr = Xr.astype(np.float)

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




################ Classification ##################

#X_train, X_test, y_train, y_test = train_test_split(Xr, y, test_size=.95, stratify=y)

# Try to change the test_size to e.g. 50 % and 99 % - how does that change the 
# effect of regularization? How does differetn runs of  test_size=.99 compare 
# to eachother?

K = 10
CV = model_selection.KFold(K, shuffle=True)
#CV = model_selection.KFold(K, shuffle=False)

# Values of lambda
lambda_interval = np.logspace(-2, 3, 10)

# Initialize variables
#T = len(lambdas)
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))


min_error_folds = np.empty(K)
opt_lambda_idx_folds = np.empty(K)
opt_lambda_fold = np.empty(K)

train_error_rate_fold = np.empty((len(lambda_interval),K))
test_error_rate_fold = np.empty((len(lambda_interval),K))

c=0
for train_index, test_index in CV.split(X,y):
    # extract training and test set for current CV fold
    X_train = Xr[train_index]
    y_train = y[train_index]
    X_test = Xr[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10 
    
    # Standardize the training and set set based on training set mean and std
    mu = np.mean(X_train, 0)
    sigma = np.std(X_train, 0)

    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma
    
    # Fit regularized logistic regression model to training data to predict 
    # the type of wine
    train_error_rate = np.zeros(len(lambda_interval))
    test_error_rate = np.zeros(len(lambda_interval))
    coefficient_norm = np.zeros(len(lambda_interval))
    for k in range(0, len(lambda_interval)):
        mdl = LogisticRegression(penalty='l2', C=1/lambda_interval[k] )
    
        mdl.fit(X_train, y_train)

        y_train_est = mdl.predict(X_train).T
        y_test_est = mdl.predict(X_test).T
    
        train_error_rate[k] = np.sum(y_train_est != y_train) / len(y_train)
        test_error_rate[k] = np.sum(y_test_est != y_test) / len(y_test)
        
        w_est = mdl.coef_[0] 
        coefficient_norm[k] = np.sqrt(np.sum(w_est**2))
    
    train_error_rate_fold[:,c] = train_error_rate
    test_error_rate_fold[:,c] = test_error_rate
    
    min_error_folds[c] = np.min(test_error_rate)
    opt_lambda_idx_folds[c] = np.argmin(test_error_rate)
    opt_lambda_fold[c] = lambda_interval[int(opt_lambda_idx_folds[c])]
    
    c+=1
    

min_error = np.min(min_error_folds)
opt_lambda_idx = np.argmin(min_error_folds)
opt_lambda = opt_lambda_fold[opt_lambda_idx]

plt.figure(figsize=(8,8))
#plt.plot(np.log10(lambda_interval), train_error_rate*100)
#plt.plot(np.log10(lambda_interval), test_error_rate*100)
#plt.plot(np.log10(opt_lambda), min_error*100, 'o')
plt.semilogx(lambda_interval, train_error_rate_fold[:,opt_lambda_idx]*100)
plt.semilogx(lambda_interval, test_error_rate_fold[:,opt_lambda_idx]*100)
plt.semilogx(opt_lambda, min_error*100, 'o')
plt.text(1e-2, 3, "Minimum test error: " + str(np.round(min_error*100,2)) + ' % at 1e' + str(np.round(np.log10(opt_lambda),2)))
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.ylabel('Error rate (%)')
plt.title('Classification error')
plt.legend(['Training error','Test error','Test minimum'],loc='upper right')
plt.ylim([0, 40])
plt.grid()
plt.show()    

plt.figure(figsize=(8,8))
plt.semilogx(lambda_interval, coefficient_norm,'k')
plt.ylabel('L2 Norm')
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.title('Parameter vector L2 norm')
plt.grid()
plt.show()








