import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn import tree
from platform import system
from os import getcwd
from toolbox_02450 import windows_graphviz_call
from matplotlib.image import imread
from sklearn.dummy import DummyClassifier
from scipy import stats
from toolbox_02450 import mcnemar


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
N, M = X.shape

#Make standadized values of Xr (subtract mean and devide by standard deviation)
Xr_stand = stats.zscore(Xr)


################ Classification ##################

K = 10 #This is for both inner and outer CV split
CV = model_selection.KFold(K, shuffle=True)

# Values of lambda
lambda_interval = np.array([0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.15,0.20])
#lambda_interval = np.logspace(-2, 3, 10)

# Values of dt depth
max_depth_range = range(2,20)

# Initialize variables
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))

all_lr_mdl = []
all_dt_mdl = []
all_baseline_mdl = []

min_test_errors_lr = np.empty((K))
min_test_errors_dt = np.empty((K))
baseline_test_error_rate = np.empty(K)
decisiontree_test_error_rate = np.empty(K)

# for CI and p values
yhat_baseline = []
yhat_lr = []
yhat_dt = []
y_true = []

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
    
    all_baseline_mdl.append(baselinemdl)
    baseline_test_error_rate[c] = np.sum((baselinemdl.predict(X_test)) != y_test) / len(y_test)
    
    yhat_baseline.append(baselinemdl.predict(X_test))
    
    ##Innerfold
    Egen_lr = np.empty((len(lambda_interval)))
    Egen_dt = np.empty((len(max_depth_range)))
    
    Internalmodel_lr = []
    Internalmodel_dt = []
    
    train_error_rate_internal_lr = np.zeros((len(lambda_interval),K))
    test_error_rate_internal_lr = np.zeros((len(lambda_interval),K))
    
    test_error_rate_internal_dt = np.zeros((len(max_depth_range),K))
    
    for (k2, (train_index_internal, test_index_internal)) in enumerate(CV.split(X_train,y_train)):
        #extract data 
        X_train_internal = Xr[train_index_internal]
        y_train_internal = y[train_index_internal]
        X_test_internal = Xr[test_index_internal]
        y_test_internal = y[test_index_internal]
    
        #Standardize data
        X_train_internal = stats.zscore(X_train_internal)
        X_test_internal = stats.zscore(X_test_internal)
        
        ##logistic regression model loop
        coefficient_norm = np.zeros(len(lambda_interval))
        count = 0
        for k in range(0, len(lambda_interval)):
            
            mdl_lr = LogisticRegression(penalty='l2', C=lambda_interval[k] )
            
            mdl_lr.fit(X_train_internal, y_train_internal)
            
            y_train_est_internal = mdl_lr.predict(X_train_internal).T
            y_test_est_internal = mdl_lr.predict(X_test_internal).T
            
            # save test and train error for each lamda
            train_error_rate_internal_lr[count][k2] = np.sum(y_train_est_internal != y_train_internal) / len(y_train_internal)
            test_error_rate_internal_lr[count][k2] = np.sum(y_test_est_internal != y_test_internal) / len(y_test_internal)
            
            Internalmodel_lr.append(mdl_lr)
            
            w_est = mdl_lr.coef_[0]
            coefficient_norm[k] = np.sqrt(np.sum(w_est**2))
            
            count += 1
        
        ## decision tree model loop
        count = 0
        for k in max_depth_range:
            dt_mdl = tree.DecisionTreeClassifier(max_depth=k)
            
            dt_mdl.fit(X=X_train_internal, y=y_train_internal)
            
            y_test_est_internal_dt = dt_mdl.predict(X_test_internal)
            
            test_error_rate_internal_dt[count][k2] = np.sum(y_test_est_internal_dt != y_test_internal) / len(y_test_internal)
            
            Internalmodel_dt.append(dt_mdl)
            
            count += 1
            
    
    # training outer model and findeing test error for logistic regression
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
    
    yhat_lr.append(y_test_est)
    
    min_test_errors_lr[c] = np.sum(y_test_est != y_test) / len(y_test)    
    
    
    # training outer model and findeing test error for decision tree
    count = 0
    for errorsmdl in test_error_rate_internal_dt:
        Egen_dt[count] = (sum(errorsmdl))*(len(test_index_internal)/len(train_index))
        count += 1

    mdl_dt = Internalmodel_dt[np.argmin(Egen_dt)]
     
    mdl_dt.fit(X_train, y_train)
    all_dt_mdl.append(mdl_dt)

    y_test_est_dt = mdl_dt.predict(X_test)
    
    yhat_dt.append(y_test_est_dt)
    
    min_test_errors_dt[c] = np.sum(y_test_est_dt != y_test) / len(y_test)
    
    y_true.append(y_test)
    
    c+=1
    

yhat_baseline = np.concatenate(yhat_baseline)
yhat_lr = np.concatenate(yhat_lr)
yhat_dt = np.concatenate(yhat_dt)
y_true = np.concatenate(y_true)


# save min error and best model
min_index_lr = np.argmin(min_test_errors_lr)
min_error_lr = np.min(min_test_errors_lr)
best_mdl_lr = all_lr_mdl[min_index_lr]

min_index_dt = np.argmin(min_test_errors_dt)
min_error_dt = np.min(min_test_errors_dt)
best_mdl_dt = all_dt_mdl[min_index_dt]

min_index_baseline = np.argmin(baseline_test_error_rate)

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

# plot the best decision tree
fname='tree_' + 'gini' + '_CHD_data'
out = tree.export_graphviz(best_mdl_dt, out_file=fname + '.gvz', feature_names=attributeNames[0:9])

if system() == 'Windows':
    # N.B.: you have to update the path_to_graphviz to reflect the position you 
    # unzipped the software in!
    path_to_graphviz = r'C:\Users\thore\.spyder-py3\Graphviz' # CHANGE THIS
    windows_graphviz_call(fname=fname,
                          cur_dir=getcwd(),
                          path_to_graphviz=path_to_graphviz)
    plt.figure(figsize=(12,12))
    plt.imshow(imread(fname + '.png'))
    plt.box('off'); plt.axis('off')
    plt.show()


### print important results
print("A logistic regression model with lamda value {0} and coefs:".format(best_mdl_lr.C))
count = 0
attributeNames[0:9]
for name in attributeNames[0:9]:
    print("Attribute {0} with weight {1}".format(name,(best_mdl_lr.coef_)[0][count]))
    count += 1


print("------------ Logistic regression ------------")
count = 0
for mdl_tmp in all_lr_mdl:
    print("lamda choosen is: {0} with test error: {1}".format('{0:.3f}'.format(mdl_tmp.C),min_test_errors_lr[count]))
    count += 1

print("------------ Decision tree ------------")
count = 0
for mdl_tmp in all_dt_mdl:
    print("Max depth is: {0} with test error: {1}".format(mdl_tmp.get_depth(),min_test_errors_dt[count]))
    count += 1

print("------------ Baseline model------------")
count = 0
for mdl_tmp in all_baseline_mdl:
    print("Baseline with test error: {0}".format(baseline_test_error_rate[count]))
    count += 1

## CI and p-value using mcnemar test
alpha = 0.05

## lr and baseline
[thetahat_lr_base, CI_lr_base, p_lr_base] = mcnemar(y_true, yhat_lr, yhat_baseline, alpha=alpha)
print("theta = theta_lr-theta_baseline point estimate", thetahat_lr_base, " CI: ", CI_lr_base, "p-value", p_lr_base)

## lr and dt
[thetahat_lr_dt, CI_lr_dt, p_lr_dt] = mcnemar(y_true, yhat_lr, yhat_dt, alpha=alpha)
print("theta = theta_lr-theta_dt point estimate", thetahat_lr_dt, " CI: ", CI_lr_dt, "p-value", p_lr_dt)

## dt and baseline
[thetahat_dt_base, CI_dt_base, p_dt_base] = mcnemar(y_true, yhat_dt, yhat_baseline, alpha=alpha)
print("theta = theta_dt-theta_baseline point estimate", thetahat_dt_base, " CI: ", CI_dt_base, "p-value", p_dt_base)




