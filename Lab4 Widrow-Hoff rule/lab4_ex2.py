#!/usr/bin/env python
import numpy as np

def trainPerceptron(w, d, learningRate, X, bias):

    #sigma = 0.01

    tr_mse = 10000
    
    while tr_mse > 0:
        for i in range(len(X)):
            y = np.dot(w[1:].T, X[i])
            w[1:] += learningRate * np.dot(d[i] - y, X[i])
            w[0] += learningRate * (d[i] - y) * bias

        temp = np.dot((d - (np.dot(X, w[1:]) + w[0])).T, (d - (np.dot(X, w[1:]) + w[0])))
        if tr_mse <= temp:
            print("Training MSE: ", tr_mse)
            print("If training MSE > 150 try to run again for better results")
            break
        else:
            tr_mse = temp

    #w_batch = np.dot(np.dot( np.linalg.inv( np.dot(X.T, X) ), X.T), d)
    #print("these are the weights in batch mode",w_batch)

    print('\nLearned everything from train data!\n')
    return w

def testPerceptron(w, X, bias):
    list_y = np.zeros((len(X),1))
    for i in range(len(X)):
        list_y[i] = 1 if np.dot(w[1:].T, X[i]) + w[0] * bias > 0 else -1

    return list_y


eta = 0.001
bias = 1
train_data = np.loadtxt('data5D_train.csv')
d = train_data[:,-1]
train_data = train_data[:,:-1]

w = np.random.random(len(train_data[0]) + 1) * 2 - 1
print('Weights before training: ', w, '\n')

w = trainPerceptron(w, d, eta, train_data, bias)
print('Weights learned through training: ', w, '\n')

test_data = np.loadtxt('data5D_test.csv')
y = testPerceptron(w, test_data, bias)
np.savetxt('output.csv', y, '%i')
print('Saved test results to output.csv')