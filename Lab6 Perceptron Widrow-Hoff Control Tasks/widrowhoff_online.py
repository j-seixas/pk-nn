#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class WidrowHoffOnline:
    """
    This is a Widrow-Hoff model which can process one example at a time.
    Can be used for both classification and regression problems
    Assumption: class label is given as {-1, 1} in classification problems
    """
    def __init__(self, num_of_inputs):
        """Constructor"""
        self.num_of_inputs = num_of_inputs
        self.w = None
        self.InitWeights()
    def InitWeights(self):
        """Initializes weights to random values"""
        self.w = -1 + 2 * np.random.rand(self.num_of_inputs + 1,)
    def Forward(self, x): 
        """Forward pass - calculate the output as a real value of the neuron for one example x"""
        y = np.dot(self.w[1:].T, x) + self.w[0]
        return y
    def ForwardClassify(self, x): 
        """
        Forward pass - calculate the output as {-1, 1} by comparing the real output value of the neuron with threshold 0; 
        for one example x
        """
        return 1 if self.Forward(x) > 0 else -1
    def Update(self, x, d, eta):
        """Calculate the output for x (one example), and update the weights"""
       
        y = self.Forward(x)
        self.w[1:] += eta * np.dot(d - y, x)
        self.w[0] += eta * (d - y)
    def Train(self, X, D, eta, epochs):
        """
        Train for the maximum number of epochs
        X: matrix with examples, each examples as a row
        D: vector of real values required for examples in rows of X 
        """
        for i in range(epochs):
            for j in range(len(X)):
                self.Update(X[j], D[j], eta)
    def CalculateErrors(self, X, D):
        """
        Calculates the number of errors - missclassifications;
        D - assumed to be {-1, 1} here
        """
        self.errors = 0
        for i in range(len(X)):
            if self.ForwardClassify(X[i]) != D[i]:
                self.errors += 1
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

perc = WidrowHoffOnline(num_of_inputs)
perc.InitWeights()

start_errors = perc.CalculateErrors(X,D)
start_mse = perc.CalculateMSE(X,D)
print('Initial number of errors=',start_errors)
print('Initial MSE=',start_mse)

print('Training...')
max_epochs = 200
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
test_ans = []
for x in test_data:
    test_ans.append ( perc.ForwardClassify(x) )
test_ans = np.array(test_ans)
print('Saving classifications for test data...')
np.savetxt('test_data_classifications_whonline.csv', test_ans)

print('Checking test error...')
true_test_labels = np.loadtxt('test10D_correct_ans.csv')
test_errors = (true_test_labels != test_ans).sum()
print('Test errors=',test_errors,' -> ',test_errors/float(test_data.shape[0]),'%')


print()
print('---REGRESSION PROBLEM---')
xmin = -6
xmax = 6
x = np.arange(xmin, xmax, 0.5)
print ('x=',x)

#real values of unknown process
a = 0.6
b = -0.4
d = a*x + b

#training data with noise (e.g., measurement errors)
sigma = 0.2
tr_d = d + np.random.randn(len(d)) * sigma

x.shape = (x.shape[0], 1)

perc_reg = WidrowHoffOnline(1)
start_mse = perc_reg.CalculateMSE(x, tr_d)
print('Initial MSE=', start_mse)

print('Training for regression...')
eta = 0.01
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