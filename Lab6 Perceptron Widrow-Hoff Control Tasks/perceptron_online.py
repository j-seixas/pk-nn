#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class PerceptronOnline:
    """
    This is a perceptron which can process one example at a time.
    Assumption: class label is given as {-1, 1}
    """
    def __init__(self, num_of_inputs):
        """Perceptron constructor"""
        self.num_of_inputs = num_of_inputs
    def InitWeights(self):
        """Initializes weights to random values"""
        self.w = -1 + 2 * np.random.rand(self.num_of_inputs,)
        self.w0 = -1 + 2 * np.random.rand()
    def Forward(self, x): 
        """Forward pass - calculate the output as {-1, 1} of the neuron for one example x"""
        y = np.dot(self.w, x) + self.w0
        return 1 if y > 0 else -1
    def Update(self, x, d, eta):
        """Calculate the output for x (one example), compare with d and update the weights if necessary"""
        if self.Forward(x) != d:
            self.w += eta * x * d
            self.w0 += eta * 1 * d
    def Train(self, X, D, eta, epochs):
        """
        Train for the maximum number of epochs or until the classification error is 0
        X: matrix with examples, each examples as a row
        D: vector of correct class labels for examples in rows of X
        The update to the weights vector is done after processing each example
        """
        for i in range(epochs):
            if self.CalculateErrors(X, D) == 0:
                break
            for j in range(len(X)):
                self.Update(X[j], D[j], eta)
    def CalculateErrors(self, X, D):
        """Calculates the number of errors - missclassifications"""
        self.errors = 0
        for i in range(len(X)):
            if self.Forward(X[i]) != D[i]:
                self.errors += 1
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

perc = PerceptronOnline(num_of_inputs)
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
test_ans = []
for x in test_data:
    test_ans.append ( perc.Forward(x) )
test_ans = np.array(test_ans)
print('Saving classifications for test data...')
np.savetxt('test_data_classifications_perconline.csv', test_ans)

print('Checking test error...')
true_test_labels = np.loadtxt('test10D_correct_ans.csv')
test_errors = (true_test_labels != test_ans).sum()
print('Test errors=',test_errors,' -> ',test_errors/float(test_data.shape[0]),'%')