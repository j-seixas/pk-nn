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
            #print('mse=',mse)
            classification_error = self.GetClassificationError(labels)
            #print('classification_error=',classification_error)
            #print()            
    def Learn(self, X, ClsIndx):
        dtrain = encode_labels_as_binary(ClsIndx, self.output_num)
        self.TrainBatch(X, dtrain, ClsIndx, self.sigma, self.eta, self.max_epochs)       

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
def generate_rbfnn(inputs_num, outputs_num):
    rbfnn_model = RBFNN(inputs_num, 20, outputs_num)
    rbfnn_model.eta = 0.1
    rbfnn_model.sigma = 0.1
    rbfnn_model.max_epochs = 200
    return rbfnn_model
############################################################################### 
def cross_validation(X, labels, model_generator, num_folds): 
    print('\nStarting cross-validation...')
    ex_num = X.shape[0] #number of examples
    inputs_num = X.shape[1]
    outputs_num = len(set(labels)) #number of classes
    #split data into num_folds parts
    indxs = np.random.randint(num_folds, size=ex_num)
    train_errors = []
    test_errors = []
    for i in range(num_folds):
        #create the current train and test sets
        trainX = X[indxs != i,:]
        train_labels = labels[indxs != i]
        testX = X[indxs == i,:]
        test_labels = labels[indxs == i]
        #get the model and train it
        print('Training model',i+1,'...')
        model = model_generator(inputs_num, outputs_num) #get a new model
        model.Learn(trainX, train_labels)
        #check the model on train data
        print('Checking the model on train data...')
        model.Forward(trainX)
        ans = model.GetPredictions()
        train_error_rate = (ans!=train_labels).sum()/trainX.shape[0]
        #check the model on test data
        print('Checking the model on test data...')
        model.Forward(testX)
        ans = model.GetPredictions()
        test_error_rate = (ans!=test_labels).sum()/testX.shape[0]
        train_errors.append(train_error_rate)
        test_errors.append(test_error_rate)
    train_errors = np.array(train_errors)
    test_errors = np.array(test_errors)
    stats = {}
    stats['train_errors'] = train_errors
    stats['test_errors'] = test_errors
    stats['train_error_mean'] = train_errors.mean()
    stats['test_error_mean'] = test_errors.mean()
    stats['train_error_std'] = train_errors.std()
    stats['test_error_std'] = test_errors.std()
    print('Cross-validation finished\n')
    return stats
###############################################################################

X = np.loadtxt('pima-diabetes.csv', delimiter=",", dtype=str)

classes = set(X[:,-1])
for clsname, clsindx in zip(classes, range(len(classes))):
    print(clsname, clsindx)
    X[X==clsname] = clsindx
labels = X[:,-1].astype('int32')
X = X[:,:-1].astype(np.float)
#print(X)
print(X.shape)
#print(labels)

xval = cross_validation(X, labels, generate_rbfnn, 10)

print('Results for cross-validation:')
for key in xval:
    print(key, xval[key],'')
