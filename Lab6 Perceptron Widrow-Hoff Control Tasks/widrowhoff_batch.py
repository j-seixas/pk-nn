#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class WidrowHoffBatch:
    """
    This is a WidrowHoff model which can process all examples at a time.
    Can be used for both classification and regression problems
    Assumption: class label is given as {-1, 1} in classification problems
    """
    def __init__(self, num_of_inputs):
        """Constructor"""
        self.num_of_inputs = num_of_inputs
        self.InitWeights()
    def InitWeights(self):
        """Initializes weights to random values"""
        self.w = -1 + 2 * np.random.rand(self.num_of_inputs + 1,)
    def Forward(self, X): 
        """
        Forward pass - calculate the output as a vector of real values of the neuron for all examples in X
        X: matrix with examples as rows
        """
        Y = np.dot(X, self.w[1:].T) + self.w[0]
        return Y
    def ForwardClassify(self, X): 
        """
        Forward pass - calculate the output as a vector of {-1, 1} by comparing the real output values of the neuron with threshold 0; 
        X: matrix with examples as rows
        """
        Y = self.Forward(X)
        Y[Y>0] = 1
        Y[Y<=0] = -1
        return Y 
    def Update(self, X, D, eta):
        """Calculate the output for all examples in X (as rows), and update the weights """
        #self.w[1:] = np.dot(np.dot( np.linalg.inv( np.dot(X.T, X) ), X.T), D)
        Y = self.Forward(X)
        self.w[1:] += eta * np.dot(X.T, (D - Y))
        self.w[0] += eta * np.dot(np.ones((1, X.shape[0])), (D-Y))
    def Train(self, X, D, eta, epochs):
        """
        Train for the maximum number of epochs
        X: matrix with examples, each examples as a row
        D: vector of real values required for examples in rows of X 
        """
        for i in range(epochs):
            if self.CalculateErrors(X, D) == 0:
                break
            
            self.Update(X, D, eta)
    def CalculateErrors(self, X, D):
        """
        Calculates the number of errors - missclassifications
        D - assumed to be {-1, 1} here
        """
        classification = self.ForwardClassify(X)
        self.errors = (classification!=D).sum()
        return self.errors
    def CalculateMSE(self, X, D):
        """
        Calculates the mean square error 
        D - assumed to be a vector of any real values here
        """
        return ( ((np.dot(X, self.w[1:]) + self.w[0]) - D) * ((np.dot(X, self.w[1:]) + self.w[0]) - D) ).sum() / len(X)
    

    
##############################################################################
#DO NOT CHANGE THE FOLLOWING CODE
##############################################################################     
print('---CLASSIFICATION PROBLEM---')
print('Loading train data...')
train_data = np.loadtxt('train10D.csv')
X = train_data[:,:-1]
D = train_data[:,-1]
num_of_inputs = X.shape[1]
print('Train data:')
print('Number of examples=',X.shape[0])
print('Number of inputs=',num_of_inputs)

perc = WidrowHoffBatch(num_of_inputs)
perc.InitWeights()

start_errors = perc.CalculateErrors(X,D)
start_mse = perc.CalculateMSE(X,D)
print('Initial number of errors=',start_errors)
print('Initial MSE=',start_mse)

print('Training...')
max_epochs = 100
eta = 0.001
perc.Train(X, D, eta, max_epochs)
print('End of training')

train_errors = perc.CalculateErrors(X,D)
train_mse = perc.CalculateMSE(X,D)
print('Errors for train data after training=',train_errors)
print('MSE for train data after training=',train_mse)

print('Loading test data...')
test_data = np.loadtxt('test10D.csv')
print('Test data:')
print('Number of examples=',test_data.shape[0])
print('Number of inputs=',test_data.shape[1])

print('Calculating answers for test data...')
test_ans = perc.ForwardClassify(test_data)
print('Saving classifications for test data...')
np.savetxt('test_data_classifications_whbatch.csv', test_ans)

print('Checking test error...')
true_test_labels = np.loadtxt('test10D_correct_ans.csv')
test_errors = (true_test_labels != test_ans).sum()
print('Test errors=',test_errors,' -> ',test_errors/float(test_data.shape[0]),'%')

print()
print('---REGRESSION PROBLEM---')
xmin = -6
xmax = 6
x = np.arange(xmin, xmax, 0.5)

#real values of unknown process
a = 0.6
b = -0.4
d = a*x + b

#training data with noise (e.g., measurement errors)
sigma = 0.2
tr_d = d + np.random.randn(len(d)) * sigma

x.shape = (x.shape[0], 1)

perc_reg = WidrowHoffBatch(1)
start_mse = perc_reg.CalculateMSE(x, tr_d)
print('Initial MSE=', start_mse)

print('Training for regression...')
eta = 0.001
max_epochs = 100
perc_reg.Train(x, tr_d, eta, max_epochs)

train_mse = perc_reg.CalculateMSE(x, tr_d)
print('After training, training MSE=', train_mse)

#test data 
x_test = np.arange(xmin, xmax, 0.3)
d_test = a*x_test + b
x_test.shape = (x_test.shape[0],1)

test_mse = perc_reg.CalculateMSE(x_test, d_test)
print('After training, testing MSE=', test_mse)