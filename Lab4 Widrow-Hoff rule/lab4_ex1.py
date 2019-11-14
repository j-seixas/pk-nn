#!/usr/bin/env python
import numpy as np

def trainPerceptron(w, d, learningRate, X, bias):

    #sigma = 0.01

    tr_mse = 10000
    not_decreasing = 0

    while tr_mse > 0:
        for i in range(len(X)):
            y = np.dot(w[1:].T, X[i]) + w[0] * bias
            w[1:] += learningRate * np.dot(d[i] - y, X[i])
            w[0] += learningRate * (d[i] - y) * bias

        temp = np.dot((d - (np.dot(X, w[1:]) + w[0])).T, (d - (np.dot(X, w[1:]) + w[0])))
        if tr_mse <= temp and not_decreasing == 3:
            print("Training MSE: ", tr_mse)
            break
        elif tr_mse <= temp:
            tr_mse = temp
            not_decreasing += 1
        else:
            tr_mse = temp
            not_decreasing = 0

    #w_batch = np.dot(np.dot( np.linalg.inv( np.dot(X.T, X) ), X.T), d)
    #print("these are the weights in batch mode",w_batch)

    print('\nLearned everything from train data!\n')
    return w

eta = 0.001
bias = 1
data = np.random.rand(7, 5) * 2 - 1
d = np.random.rand(len(data)) * 2 - 1
w = np.random.random(len(data[0]) + 1) * 2 - 1
print('Weights before training: ', w, '\n')

w = trainPerceptron(w, d, eta, data, bias)
print('Weights learned through training: ', w, '\n')
