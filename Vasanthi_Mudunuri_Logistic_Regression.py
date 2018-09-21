import numpy as np
import math
from scipy.optimize import minimize

def flower_to_float(s):
    d = {'Iris-setosa':0.,'Iris-versicolor':1.,'Iris-virginica':2.}
    return d[s]
irises = np.loadtxt('C:\iris.txt',delimiter=',',converters={4:flower_to_float})    
Data=irises[:,0:4]
Label=irises[:,4]
Bias=np.reshape((np.ones(150, dtype=int)),(150,1))
Data=np.concatenate((Bias,Data),axis=1)
#print Data
Labels=np.ones((150,3),dtype=int)
Labels[0:50,1:3]=0
Labels[50:100,(0,2)]=0
Labels[100:150,0:2]=0
#print Labels
                                               
TrainingData=Data[1::2]
TestData=Data[::2]
#print TrainingData
#print TestData

TrainingLabels=Labels[1::2]
TestLabels=Labels[::2]
#print TrainingLabels
#print TestLabels

def logistic_regression(w):
    alpha=np.logspace(-8,0,100)
    alpha=alpha[19]
    a=np.dot((np.dot(np.transpose(w),w)),alpha/2)
    Result=np.array([])
    for n in range(0,TrainingData.shape[0]):
        tn0=TrainingLabels[n][0]
        tn1=TrainingLabels[n][1]
        tn2=TrainingLabels[n][2]
        w1=w[0:5]
        w2=w[5:10]
        w3=w[10:15]
        Fi=TrainingData[n]
        b1=np.dot(np.dot(np.transpose(w1),Fi),tn0)
        b2=np.dot(np.dot(np.transpose(w2),Fi),tn1)
        b3=np.dot(np.dot(np.transpose(w3),Fi),tn2)
        b=b1+b2+b3
        #print b
        c1=math.exp(np.dot(np.transpose(w1),Fi))
        c2=math.exp(np.dot(np.transpose(w2),Fi))
        c3=math.exp(np.dot(np.transpose(w3),Fi))
        c=math.log(c1+c2+c3)
        #print c
        difference=b-c
        result=a-difference
        Result=np.append(Result,result)
    return np.sum(Result)
           

if __name__ == '__main__':        
    w_init=np.ones(15)
    w_hat=minimize(logistic_regression,w_init).x
    print w_hat
    Prediction=np.array([])
    for n in range(0,TestData.shape[0]):
        Fi=TestData[n]
        w1=w_hat[0:5]
        w2=w_hat[5:10]
        w3=w_hat[10:15]
        z1=math.exp(np.dot(np.transpose(w1),Fi))
        print z1
        z2=math.exp(np.dot(np.transpose(w2),Fi))
        print z2
        z3=math.exp(np.dot(np.transpose(w3),Fi))
        print z3
        softmax1=z1/(z1+z2+z3)
        softmax2=z2/(z1+z2+z3)
        softmax3=z3/(z1+z2+z3)
        classification=np.array([softmax1,softmax2,softmax3])
        print classification
        prediction=np.argmax(classification)
        print prediction
        Prediction=np.append(Prediction,prediction)
    print Prediction     
    print Label[::2]
    print "Accuracy: ", ((Prediction == Label[::2]).mean())*(100.0),'%'
