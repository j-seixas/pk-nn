#!/usr/bin/env python
import numpy as np

def trainPerceptron(w, w0, bias, learningRate, data, max_iter):
    hasErrors = True
    while hasErrors and (max_iter > 0 or max_iter == -5):
        i = 0
        hasErrors = False

        for t in data:
            y = np.dot(w, t[0]) + w0

            u = 1 if y > 0 else -1
                
            if u != t[1]:
                hasErrors = True
                w = [w_i + learningRate * t0_i * t[1] for w_i, t0_i in zip(w, t[0])]
                w0 = w0 + learningRate * bias * t[1]
            
            i += 1
        max_iter -= 1

    print('\nLearned everything from train data!\n')
    return w,w0


def testDataWithPerceptron(w, w0, bias, learningRate, data):
    f = open('output.csv','w')
    i = 0
    for t in data:
        y = np.dot(w, t) + w0

        u = 1 if y > 0 else -1
        f.write('{}\n'.format(u))
                
        i += 1

    f.close()
    print('\nProcessed everything from test data!\n')

    
train_file = open('data5D_train.csv','r')
lines = train_file.readlines()
train_file.close()

trainData = []
for l in lines:
    nums = [float(v) for v in l.split()]
    trainData.append([np.array(nums[:-1]), nums[-1]])

w, w0 = trainPerceptron(np.array(np.random.random(len(trainData[0][0])) * 2 - 1), 0.1, 1, 0.1, trainData, -5)
print('Weights learned through training: ', w, w0)

test_file = open('data5D_test.csv','r')
lines = test_file.readlines()
test_file.close()

testData = []
for l in lines:
    nums = [float(v) for v in l.split()]
    testData.append(np.array(nums))
testDataWithPerceptron(w, w0, 1, 0.1, testData)

print('Used the weights: ', w, w0)