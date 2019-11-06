#!/usr/bin/env python
import numpy as np
#import matplotlib.cm as cm
#import matplotlib.mlab as mlab
#import matplotlib.pyplot as plt
#import matplotlib.animation as animation

#%matplotlib notebook

def trainPerceptron(w, d, learningRate, X):

    tr_mse = 10000
    while tr_mse > 0:
        for i in range(len(X)):
            w += learningRate * np.dot(d[i] - np.dot(w.T, X[i]), X[i])

        temp = np.dot((d - np.dot(X, w)).T, (d - np.dot(X, w)))
        if tr_mse <= temp:
            print("Online mode Training MSE: ", tr_mse)
            break
        else:
            tr_mse = temp

    print('\nLearned everything from train data!\n')
    return w


xmin = -6
xmax = 6
#ymin = -6
#ymax = 6

x = np.arange(xmin, xmax, 0.5)
print ('x=',x)

#real values of unknown process
a = 0.6
b = -0.4
d = a*x + b

#training data with noise (e.g., measurement errors)
sigma = 0.2
tr_d = d + np.random.randn(len(d)) * sigma

#plt.xlim(xmin,xmax)
#plt.ylim(ymin,ymax)

#plt.plot(x, d)
#plt.plot(x, tr_d, 'o')

#we add the column with "1" values directly here - not efficient in general
X = np.vstack((x, np.ones(len(tr_d)))).T
print('X=',X)
print(X.shape)


#weights for the neuron - there is no iterative process here!
w = np.dot(np.dot( np.linalg.inv( np.dot(X.T, X) ), X.T), tr_d)
print('Batch mode weights: w=', w)

w = np.random.random(len(X[0])) * 2 - 1
print('Online training weights before training: ', w, '\n')

eta = 0.001
w = trainPerceptron(w, d, eta, X)
print('Weights learned through training: ', w, '\n')

#neuron responses
y = w[0]*x + w[1]
#plt.plot(x, y, 'r')
#plt.plot(x, y, 'rx')

#training error
t_mse = np.dot((y - tr_d).T, (y - tr_d))
print('training mse = ', t_mse)

#sample some new points as test data
x_test = np.arange(xmin, xmax, 0.3)
d_test = a*x_test + b
y_test = w[0]*x_test + w[1]
test_mse = np.dot((y_test - d_test).T, (y_test - d_test))
print('testing mse = ', test_mse)

#plt.figure()
#plt.plot(x_test, d_test)
#plt.plot(x_test, y_test,'rx')

#plt.show()