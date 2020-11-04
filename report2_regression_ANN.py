import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd
from matplotlib.pyplot import (figure, plot, title, xlabel, ylabel, show,
                               legend, subplot, xticks, yticks, boxplot, hist,
                               ylim)
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate
import torch
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats

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
###### Part A
Ysbp = (Xall[:,0]).reshape(462,1) #Sbp
Xdt = Xall[:,1:10]


# Normalize data
Xdt = stats.zscore(Xdt)

# Parameters for neural network classifier
n_hidden_units = 30      # number of hidden units
n_replicates = 1           # number of networks trained in each k-fold
max_iter = 40000

n_hidden_units_all = range(10, 20)

# K-fold crossvalidation
K = 3                   # only three folds to speed up this example
K_internal = 10
CV = model_selection.KFold(K, shuffle=True)

# Setup figure for display of learning curves and error rates in fold
summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))
# Make a list for storing assigned color of learning curve for up to K=10
color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']
# Define the model
allmodels = []


model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                    torch.nn.Tanh(),   # 1st transfer function,
                    torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                    # no final tranfer function, i.e. "linear output"
                    )
loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss

print('Training model of type:\n\n{}\n'.format(str(model())))
errors = [] # make a list for storing generalizaition error in each loop
for (k, (train_index, test_index)) in enumerate(CV.split(Xdt,Ysbp)): 
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
    
    # Extract training and test set for current CV fold, convert to tensors
    X_train = torch.Tensor(Xdt[train_index,:])
    y_train = torch.Tensor(Ysbp[train_index])
    X_test = torch.Tensor(Xdt[test_index,:])
    y_test = torch.Tensor(Ysbp[test_index])
    
    
    Internalerrors = np.empty((K_internal))
    Internalmodel = []
    
    ##### internal cross validation #########
    for h in n_hidden_units_all: 
        model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, h), #M features to n_hidden_units
                    torch.nn.Tanh(),   # 1st transfer function,
                    torch.nn.Linear(h, 1), # n_hidden_units to 1 output neuron
                    # no final tranfer function, i.e. "linear output"
                    )
        
        # Train the net on training data
        net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
    
        print('\n\tBest loss: {}\n'.format(final_loss))
        
        # Determine estimated class labels for test set
        y_test_est = net(X_test)
    
        # Determine errors and errors
        se = (y_test_est.float()-y_test.float())**2 # squared error
        mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
        
        Internalerrors[h] = (mse)
        Internalmodel.append(net)
        
        
        
    minIndex = np.argmin(Internalerrors)
    errors.append(Internalerrors[minIndex]) # store error rate for current CV fold 
    allmodels.append(Internalmodel[minIndex])
    
    
    # Display the learning curve for the best net in the current fold
    h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
    h.set_label('CV fold {0}'.format(k+1))
    summaries_axes[0].set_xlabel('Iterations')
    summaries_axes[0].set_xlim((0, max_iter))
    summaries_axes[0].set_ylabel('Loss')
    summaries_axes[0].set_title('Learning curves')

# Display the MSE across folds
summaries_axes[1].bar(np.arange(1, K+1), np.squeeze(np.asarray(errors)), color=color_list)
summaries_axes[1].set_xlabel('Fold')
summaries_axes[1].set_xticks(np.arange(1, K+1))
summaries_axes[1].set_ylabel('MSE')
summaries_axes[1].set_title('Test mean-squared-error')
    
print('Diagram of best neural net in last fold:')
weights = [net[i].weight.data.numpy().T for i in [0,2]]
biases = [net[i].bias.data.numpy() for i in [0,2]]
tf =  [str(net[i]) for i in [1,2]]

attributeNames2 = [name[:] for name in attributeNames[1:10]]

draw_neural_net(weights, biases, tf, attribute_names=attributeNames2)

# Print the average classification error rate
print('\nEstimated generalization error, RMSE: {0}'.format(round(np.sqrt(np.mean(errors)), 4)))

# When dealing with regression outputs, a simple way of looking at the quality
# of predictions visually is by plotting the estimated value as a function of 
# the true/known value - these values should all be along a straight line "y=x", 
# and if the points are above the line, the model overestimates, whereas if the
# points are below the y=x line, then the model underestimates the value
plt.figure(figsize=(10,10))
y_est = y_test_est.data.numpy().reshape(154,1); y_true = y_test.data.numpy()
axis_range = [np.min([y_est, y_true])-1,np.max([y_est, y_true])+1]
plt.plot(axis_range,axis_range,'k--')
plt.plot(y_true, y_est,'ob',alpha=.25)
plt.legend(['Perfect estimation','Model estimations'])
plt.title('Alcohol content: estimated versus true value (for last CV-fold)')
plt.ylim(axis_range); plt.xlim(axis_range)
plt.xlabel('True value')
plt.ylabel('Estimated value')
plt.grid()

plt.show()



