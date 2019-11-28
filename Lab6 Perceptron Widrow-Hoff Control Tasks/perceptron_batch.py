#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class PerceptronBatch:
    """
    This is a perceptron which can process all examples at a time.
    Assumption: class label is given as {-1, 1}
    """
    def __init__(self, num_of_inputs):
        """Perceptron constructor"""
        self.num_of_inputs = num_of_inputs
    def InitWeights(self):
        """Initializes weights to random values"""
        self.w = -1 + 2 * np.random.rand(self.num_of_inputs,)
        self.w0 = -1 + 2 * np.random.rand()
    def Forward(self, X): 
        """
        Forward pass - calculate the output as a vector of {-1, 1} of the neuron for all examples in X
        X: matrix with examples as rows
        """
        Y = np.dot(X, self.w) + self.w0
        return np.array([1 if y > 0 else -1 for y in Y])
    def Update(self, X, D, eta):
        """Calculate the output for all examples in X (as rows), compare with D and update the weights if necessary"""
        Y = self.Forward(X)
        diff = Y!=D
        updateW = np.array([X[i]*D[i] for i in range(len(Y)) if Y[i] != D[i]]).sum()
        updateW0 = np.array([D[i] for i in range(len(Y)) if Y[i] != D[i]]).sum()
        self.w += eta * (X[diff] * D[diff, None]).sum(axis=0)
        self.w0 += eta * (D[diff, None]).sum()


    def Train(self, X, D, eta, epochs):
        """
        Train for the maximum number of epochs or until the classification error is 0
        X: matrix with examples, each examples as a row
        D: vector of correct class labels for examples in rows of X
        The update to the weights vector is done once per epoch, based on all examples
        """
        for i in range(epochs):
            if self.CalculateErrors(X, D) == 0:
                break
            self.Update(X, D, eta)
    def CalculateErrors(self, X, D):
        """Calculates the number of errors - missclassifications"""
        Y = self.Forward(X)
        self.errors = len([Y[i] for i in range(len(Y)) if Y[i] != D[i]])
        return self.errors
    

##############################################################################
#DO NOT CHANGE THE FOLLOWING CODE
##############################################################################
print('Loading train data...')
train_data = np.loadtxt('train10D.csv')
X = train_data[:,:-1]
D = train_data[:,-1]
num_of_inputs = X.shape[1]
print('Train data:')
print('Number of examples=',X.shape[0])
print('Number of inputs=',num_of_inputs)

perc = PerceptronBatch(num_of_inputs)
perc.InitWeights()

start_errors = perc.CalculateErrors(X,D)
print('Initial number of errors=',start_errors)

print('Training...')
max_epochs = 100
eta = 0.01
perc.Train(X, D, eta, max_epochs)
print('End of training')

train_errors = perc.CalculateErrors(X,D)
print('Errors for train data after training=',train_errors)

print('Loading test data...')
test_data = np.loadtxt('test10D.csv')
print('Test data:')
print('Number of examples=',test_data.shape[0])
print('Number of inputs=',test_data.shape[1])

print('Calculating answers for test data...')
test_ans = perc.Forward(test_data)
print('Saving classifications for test data...')
np.savetxt('test_data_classifications_percbatch.csv', test_ans)

print('Checking test error...')
true_test_labels = np.loadtxt('test10D_correct_ans.csv')
test_errors = (true_test_labels != test_ans).sum()
print('Test errors=',test_errors,' -> ',test_errors/float(test_data.shape[0]),'%')