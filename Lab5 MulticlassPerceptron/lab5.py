#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class MCPerceptron:
    def __init__(self, num_of_classes, num_of_inputs):
        self.w =  -1 + 2 * np.random.rand(num_of_inputs, num_of_classes)  #neurons' weights as columns
        self.b = np.zeros(num_of_classes) #biases from all neurons
        self.outs = None
    def Forward(self, X):
        self.outs = np.dot(X, self.w) + self.b
        return self.outs
    def GetPredictions(self):
        return np.argmax(self.outs, axis = 1)
    def GetMSE(self, d):
        self.mse = np.linalg.norm(self.outs - d, axis = 1).sum() / d.shape[0]
        return self.mse
    def GetClassificationError(self, labels):
        return np.sum(labels != self.GetPredictions())
    def Train(self, X, d): #matrix pseudo-inverse
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), d)
        print('w=', w)
        self.w = w[1:,:]
        self.b = w[0,:]
        print('w=', self.w)
        print('b=', self.b)
    def TrainIterative(self, X, d, labels, eta, max_iters):
        self.mse_stats = []
        Y = self.Forward(X)
        for i in range(max_iters):
            self.w += eta * np.dot(X.T, d - self.outs)
            self.b += eta * np.dot(np.ones((1, X.shape[0])), d - self.outs).flatten()
            Y = self.Forward(X)
            mse = self.GetMSE(d)
            self.mse_stats.append(mse)
            #print('mse=',mse)
            #classification_error = self.GetClassificationError(labels)
            #print('classification_error=',classification_error)
            #print()

###########################################################################################################
##########################################################################
def encode_labels_as_binary(d, num_of_classes):
    rows = d.shape[0]
    labels = -1 * np.ones((rows, num_of_classes), dtype='float32')
    labels[np.arange(rows), d.T] = 1
    return labels

def encode_str_labels(d):
    labels = list(set(d))
    for i in range(len(d)):
        d[i] = labels.index(d[i])
    return d

##########################################################################
#load data
##########################################################################
X = np.loadtxt('iris.csv', dtype=str)
d = X[:,-1]
X = X[:,:-1].astype('float')
#print('X=', X)

### Enconde the string labels into numbers
d = encode_str_labels(d).astype('int')
#print('d=', d)

### Normalize the data
X = X / np.amax(X)

num_of_cls = len(set(d))
num_of_ins = X.shape[1]

### Encode the labels to binary
dtrain = encode_labels_as_binary(d, num_of_cls)
#print('dtrain=', dtrain)

p = MCPerceptron(num_of_cls, num_of_ins)
Y = p.Forward(X)

### Before Training
predictions = p.GetPredictions()
print('Predictions =',predictions)
print('MSE =', p.GetMSE(dtrain))
print('Classification errors =', np.sum(d != predictions))

##########################################################################
# Train method (uncomment the one you want and comment the other)
##########################################################################
### Pseudo-inverse
#p.Train(X, dtrain)
### Iterative
p.TrainIterative(X, dtrain, d, 0.005, 200)

### After training
Y = p.Forward(X)
#print('Y=',Y)
predictions = p.GetPredictions()
print('Predictions=', predictions)
print('MSE=', p.GetMSE(dtrain))
print('Classification errors=', np.sum(d != predictions))
print('w=', p.w)
print('b=', p.b)

plt.figure()
plt.plot(p.mse_stats)
plt.title('Training MSE')
