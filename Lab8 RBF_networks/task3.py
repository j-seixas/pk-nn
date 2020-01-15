#!/usr/bin/env python
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.animation as animation


###############################################################################
class RBFNN:
    def __init__(self, inputs_num, hidden_num, output_num):#hidden_num=number of radial neurons in the hidden layer
        self.inputs_num = inputs_num
        self.hidden_num = hidden_num
        self.output_num = output_num
        self.hcenters = np.zeros((hidden_num, inputs_num)) #centres of radial functions in the hidden layer
        self.hsigmas = np.ones(hidden_num)#sigma values of radial functions in the hidden layer
        self.outweights = np.random.rand(hidden_num, output_num) #each output neuron as a column
        self.outbiases = np.random.rand(output_num)#biases of the output linear neurons
        self.houtputs = None #outputs of radial neurons (hidden layer)
        self.netoutputs = None #output of the network (linear neurons)
        self.stats = None #statistics about the MSE during batch training
    def Print(self):#print basic info about the network
        print('hcenters:\n',self.hcenters)
        print('hsigmas:\n',self.hsigmas)
        print('outweights:\n', self.outweights)
        print('outbiases:\n',self.outbiases)        
        if self.houtputs is not None:
            print('houtputs:\n',self.houtputs)
        if self.netoutputs is not None:
            print('netoutputs:\n',self.netoutputs)  
    def Forward(self, inputs):
        ##outputs of radial neurons (hidden layer)
        self.houtputs = np.empty((inputs.shape[0], self.hcenters.shape[0]), dtype = float)
        for i in range(inputs.shape[0]): #for each training example
            self.houtputs[i,:] = np.exp(-np.sum((self.hcenters - inputs[i,:])**2, axis=1)/self.hsigmas**2)
        ##outputs of linear neurons (output layer)
        self.netoutputs = np.dot(self.houtputs, self.outweights) + self.outbiases
    def GetOutputs(self):#returns real valued outputs
        return self.netoutputs
    def GetPredictions(self):#returns class labels as 0,1,2,...
        return np.argmax(self.netoutputs, axis=1)
    def GetClassificationError(self, labels):
        return np.sum(labels!=self.GetPredictions())  
    def GetMSE(self, d):
        self.mse = ((self.netoutputs - d)*(self.netoutputs - d)).sum(axis=1).sum() /d.shape[0]
        return self.mse       
    def GetMaxRadialValue(self, X):#helper function for vizualization; for each example (row in X) returns the maximum value of any of the radial functions
        self.Forward(X)
        return self.houtputs.max(axis=1)
    def InitCenters(self, inputs, sigma):#randomly select a self.hidden_num number of training examples and copy their positions as centres of rbf neurons
        self.hsigmas = np.ones(self.hidden_num)*sigma
        indxs = set()
        while len(indxs) < self.hcenters.shape[0]:
            indxs.add(np.random.randint(0,inputs.shape[0]))
        self.hcenters = inputs[np.asarray(list(indxs)), :].copy()
    def TrainMPInv(self, X, d, sigma): #matrix pseudo inverse
        self.InitCenters(X, sigma)
        self.Forward(X)
        #now the matrix pseudoinverse for the weights of the output linear neurons
        r = np.hstack((np.ones((self.houtputs.shape[0], 1)), self.houtputs))
        w = np.dot(np.dot( np.linalg.inv( np.dot(r.T, r) ), r.T), d)
        self.w = w[1:,:]
        self.b = w[0,:]
    def TrainBatch(self, X, d, labels, sigma, eta, max_iters): #Widrow-Hoff model, delta rule
        self.InitCenters(X, sigma)
        self.Forward(X)
        self.stats = []
        for i in range(max_iters):
            self.outweights += eta*np.dot(self.houtputs.T, d - self.netoutputs)/X.shape[0]
            self.outbiases += eta*np.dot(np.ones((1,self.houtputs.shape[0])), d - self.netoutputs).flatten()/X.shape[0]
            self.Forward(X)
            mse = self.GetMSE(d)
            self.stats.append(mse)
            print('mse=',mse)
            classification_error = self.GetClassificationError(labels)
            print('classification_error=',classification_error)
            print()  
###############################################################################        
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
###############################################################################
###############################################################################        
def encode_labels_as_binary(d, num_of_classes):
    rows = d.shape[0]
    labels = -1*np.ones((rows, num_of_classes), dtype='float32')
    labels[np.arange(rows),d.T] = 1
    return labels

def encode_str_labels(d):
    labels = list(set(d))
    for i in range(len(d)):
        d[i] = labels.index(d[i])
    return d
###############################################################################        
X = np.loadtxt('pima-diabetes.csv', delimiter=",", dtype=str)
d = X[:,-1].astype('int')
X = X[:,:-1].astype('float')

### Normalize data
X = X / np.amax(np.abs(X), axis=0)
print('X=',X)


num_of_cls = len(set(d))
num_of_ins = X.shape[1]

print('num_of_cls=',num_of_cls)
print('num_of_ins=',num_of_ins)

dtrain = encode_labels_as_binary(d, num_of_cls)
#print('dtrain=',dtrain)

#experiment with the values of hidden_num and sigma, so that the training data is well covered by radial responses
hidden_num = 16 #experiment with this value
sigma = 0.5 #experiment with this value

net = RBFNN(num_of_ins, hidden_num, num_of_cls)
#net.Print()
net.Forward(X)
#net.Print()
print('MSE before training=',net.GetMSE(dtrain))
print('Classification error before training=',net.GetClassificationError(d))

net.TrainBatch(X, dtrain, d, sigma, 0.3, 600)
#net.TrainMPInv(X, d, sigma)

net.Forward(X)
#net.Print()
print('MSE after training=',net.GetMSE(dtrain))
print('Classification error after training=',net.GetClassificationError(d))
print('houts max=',net.houtputs.max(axis=1).min())
print('out w max=',net.outweights.max())
print('out w min=',net.outweights.min())


print('##############################')
print('##############################')
print('-------- MCPerceptron --------')
print('##############################')

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
p.TrainIterative(X, dtrain, d, 0.001, 200)

### After training
Y = p.Forward(X)
#print('Y=',Y)
predictions = p.GetPredictions()
print('Predictions=', predictions)
print('MSE=', p.GetMSE(dtrain))
print('Classification errors=', np.sum(d != predictions))
print('w=', p.w)
print('b=', p.b)