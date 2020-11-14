import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from toolbox_02450 import rocplot, confmatplot
from sklearn import model_selection
from toolbox_02450 import rlr_validate
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from platform import system
from os import getcwd
from toolbox_02450 import windows_graphviz_call
from matplotlib.image import imread
from sklearn import metrics
from sklearn.dummy import DummyClassifier
from scipy import stats

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
Xr_stand = stats.zscore(Xr)




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
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))

baseline_test_error_rate = np.empty(K)

opt_lambda_idx_folds = np.empty(K)
opt_lambda_fold_internal = np.empty(K)
opt_lambda_fold = np.empty(K)
all_lr_mdl = []

# train_error_rate_fold = np.empty((K))
# test_error_rate_fold = np.empty((K))
train_error_rate_fold = np.empty((len(lambda_interval),K))
test_error_rate_fold = np.empty((len(lambda_interval),K))
min_test_errors_lr = np.empty((K))

parameters = {'max_depth':range(2,20)}
criterion='gini'

decisiontree_test_error_rate = np.empty(K)

lr_mdls = []
dt_mdls = []


c=0
for train_index, test_index in CV.split(Xr,y):
    # extract training and test set for current CV fold
    X_train = Xr[train_index]
    y_train = y[train_index]
    X_test = Xr[test_index]
    y_test = y[test_index]
    
    # Standardize the training and set set based on training set mean and std
    mu = np.mean(X_train, 0)
    sigma = np.std(X_train, 0)

    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma
    
    ## Baseline model here
    baselinemdl = DummyClassifier(strategy='uniform', random_state=1)
    # fit model
    baselinemdl.fit(X_train, y_train)
    
    baseline_test_error_rate[c] = np.sum((baselinemdl.predict(X_test)) != y_test) / len(y_test)
    
    Egen_lr = np.empty((len(lambda_interval)))
    
    Internalmodel_lr = []
    
    ##Innerfold
    train_error_rate_internal_lr = np.zeros((len(lambda_interval),K))
    test_error_rate_internal_lr = np.zeros((len(lambda_interval),K))
    
    for (k2, (train_index_internal, test_index_internal)) in enumerate(CV.split(X_train,y_train)):
        #extract data 
        X_train_internal = Xr[train_index_internal]
        y_train_internal = y[train_index_internal]
        X_test_internal = Xr[test_index_internal]
        y_test_internal = y[test_index_internal]
    
        #Standardize data
        X_train_internal = stats.zscore(X_train_internal)
        X_test_internal = stats.zscore(X_test_internal)
        
        ##logistic regression
        # Fit regularized logistic regression model to training data to predict 
        # the type of wine
        internal_lr_mdls = []
        coefficient_norm = np.zeros(len(lambda_interval))
        count = 0
        for k in range(0, len(lambda_interval)):
            
            ## Logistic regression 
            mdl_lr = LogisticRegression(penalty='l2', C=1/lambda_interval[k] )
            
            mdl_lr.fit(X_train_internal, y_train_internal)
            
            y_train_est_internal = mdl_lr.predict(X_train_internal).T
            y_test_est_internal = mdl_lr.predict(X_test_internal).T
            
            # save test and train error for each lamda
            train_error_rate_internal_lr[count][k2] = np.sum(y_train_est_internal != y_train_internal) / len(y_train_internal)
            test_error_rate_internal_lr[count][k2] = np.sum(y_test_est_internal != y_test_internal) / len(y_test_internal)
            
            Internalmodel_lr.append(mdl_lr)
            
            w_est = mdl_lr.coef_[0]
            coefficient_norm[k] = np.sqrt(np.sum(w_est**2))
            
            ## DT here
            
            
            count += 1

    
    count = 0
    for errorsmdl in test_error_rate_internal_lr:
        Egen_lr[count] = (sum(errorsmdl))*(len(test_index_internal)/len(train_index))
        count += 1
        
    minIndex = np.argmin(Egen_lr)

    mdl_lr = Internalmodel_lr[minIndex]
     
    mdl_lr.fit(X_train, y_train)
    all_lr_mdl.append(mdl_lr)
    
    y_train_est = mdl_lr.predict(X_train).T
    y_test_est = mdl_lr.predict(X_test).T
    
    min_test_errors_lr[c] = np.sum(y_test_est != y_test) / len(y_test)    

    
    ##decisionstree
    clf = GridSearchCV(tree.DecisionTreeClassifier(criterion=criterion), parameters)
    clf.fit(X=X_train, y=y_train)
    tree_model = clf.best_estimator_
    print (clf.best_score_, clf.best_params_)
    
    y_test_est = tree_model.predict(X_test)
    
    # dtc = dtc.fit(X_train,y_train)
    # y_test_est = dtc.predict(y_test)
    decisiontree_test_error_rate[c] = np.sum(y_test_est != y_test) / len(y_test)
    
    c+=1
    

# save min error and best model
min_error_lr = np.min(min_test_errors_lr)
best_mdl_lr = all_lr_mdl[np.argmin(min_test_errors_lr)]


# plot for the last internal fold
min_error = np.min(test_error_rate_internal_lr[:,3])
opt_lambda_idx = np.argmin(test_error_rate_internal_lr[:,3])
opt_lambda = lambda_interval[opt_lambda_idx]


plt.figure(figsize=(8,8))
plt.semilogx(lambda_interval, train_error_rate_internal_lr[:,3]*100)
plt.semilogx(lambda_interval, test_error_rate_internal_lr[:,3]*100)
plt.semilogx(opt_lambda, min_error*100, 'o')
plt.text(1e-2, 3, "Minimum validation error: " + str(np.round(min_error*100,2)) + ' % at 1e' + str(np.round(np.log10(opt_lambda),2)))
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.ylabel('Error rate (%)')
plt.title('Classification error for last internal fold')
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

### Decision tree
# parameters = {'max_depth':range(2,20)}


# decisiontree_test_error_rate = np.empty(K)

# c=0
# for train_index, test_index in CV.split(X,y):
#     # extract training and test set for current CV fold
#     X_train = Xr_stand[train_index]
#     y_train = y[train_index]
#     X_test = Xr_stand[test_index]
#     y_test = y[test_index]
    
#     #How do we make the innner
#     criterion='gini'
#     clf = GridSearchCV(tree.DecisionTreeClassifier(criterion=criterion), parameters)
#     clf.fit(X=X_train, y=y_train)
#     tree_model = clf.best_estimator_
#     print (clf.best_score_, clf.best_params_)
    
#     y_test_est = tree_model.predict(X_test)
    
#     # dtc = dtc.fit(X_train,y_train)
#     # y_test_est = dtc.predict(y_test)
#     decisiontree_test_error_rate[c] = np.sum(y_test_est != y_test) / len(y_test)
    
#     c = c+1;
    


### plot a decision tree
# fname='tree_' + criterion + '_CHD_data'
# # Export tree graph .gvz file to parse to graphviz
# out = tree.export_graphviz(dtc, out_file=fname + '.gvz', feature_names=attributeNames[1:10])

# if system() == 'Windows':
#     # N.B.: you have to update the path_to_graphviz to reflect the position you 
#     # unzipped the software in!
#     path_to_graphviz = r'C:\Users\thore\.spyder-py3\Graphviz' # CHANGE THIS
#     windows_graphviz_call(fname=fname,
#                           cur_dir=getcwd(),
#                           path_to_graphviz=path_to_graphviz)
#     plt.figure(figsize=(12,12))
#     plt.imshow(imread(fname + '.png'))
#     plt.box('off'); plt.axis('off')
#     plt.show()











