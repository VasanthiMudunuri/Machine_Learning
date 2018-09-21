import numpy as np
import math
import matplotlib.pyplot as mplot

def RMSE(prediction,actual):
    return np.sqrt(((prediction - actual) ** 2).mean())
    
def KFold(X, Y):
    Totaldata = len(X) /5 
    for k in range(1,6):
        if k!=5:
            X_Training = np.append(X[:k * Totaldata],X[(k + 1) * Totaldata:])
            X_Validation = X[k * Totaldata:][:Totaldata]
            Y_Training = np.append(Y[:k * Totaldata],Y[(k + 1) * Totaldata:])
            Y_Validation = Y[k * Totaldata:][:Totaldata]
            yield X_Training, Y_Training, X_Validation, Y_Validation
        else:
            X_Training = X[15:90]
            X_Validation = X[15:][:Totaldata]
            Y_Training = Y[15:90]
            Y_Validation = Y[15:][:Totaldata]
            yield X_Training, Y_Training, X_Validation, Y_Validation

Data=np.loadtxt('C:\crash.txt')
x=np.array(Data[:,0])
y=np.array(Data[:,1])
Normalizedx=(x-min(x))/(max(x)-min(x))
Normalizedy=(y-min(y))/(max(y)-min(y))
Normalizedstd=(20-min(y))/(max(y)-min(y))
Beta=1/(Normalizedstd*Normalizedstd) 
sigma1=np.logspace(-2,-1,100)
print sigma1[88],sigma1[70]
RootMeanSquareError_Training_Average=np.array([])
RootMeanSquareError_Validation_Average=np.array([])
count=0
for i in sigma1:
    RootMeanSquareError_Training=np.array([])
    RootMeanSquareError_Validation=np.array([])
    for x_Training, y_Training, x_Validation, y_Validation in KFold(Normalizedx, Normalizedy):
        K1=np.array([])
        for j in x_Training:
            for k in x_Training:
                kernel1=math.exp(-((j-k)*(j-k))/(2*((i)*(i))))
                K1=np.append(K1,kernel1)        
        GramMatrix=np.reshape(K1,(x_Training.size,x_Training.size)) 
        I=np.identity(x_Training.size)
        C=np.add(GramMatrix,np.dot((1/Beta),I))     
        a=np.dot(np.linalg.inv(C),y_Training)
        y_prediction=np.array([])
        for m in range(0,x_Training.size):
            summation=0
            for n in range(0,x_Training.size):
                summation=summation+(a[n]*GramMatrix[m][n])
            y_prediction=np.append(y_prediction,summation) 
        RootMeanSquareError_Training=np.append(RootMeanSquareError_Training,RMSE(y_prediction,y_Training))  
        if count==88:
            print y_prediction.shape     
            print y_prediction
            mplot.plot(x_Training,y_Training)
            mplot.plot(x_Training,y_prediction)
            mplot.show()
        K1=np.array([])
        for j in x_Validation:
            for k in x_Validation:
                kernel1=math.exp(-((j-k)*(j-k))/(2*((i)*(i))))
                K1=np.append(K1,kernel1)        
        GramMatrix=np.reshape(K1,(x_Validation.size,x_Validation.size)) 
        I=np.identity(x_Validation.size)
        C=np.add(GramMatrix,np.dot((1/Beta),I))     
        a=np.dot(np.linalg.inv(C),y_Validation)
        y_prediction=np.array([])
        for m in range(0,x_Validation.size):
            summation=0
            for n in range(0,x_Validation.size):
                summation=summation+(a[n]*GramMatrix[m][n])
            y_prediction=np.append(y_prediction,summation)  
        RootMeanSquareError_Validation=np.append(RootMeanSquareError_Validation,RMSE(y_prediction,y_Validation)) 
        if count==70:
            print y_prediction.shape     
            print y_prediction
            mplot.plot(x_Validation,y_Validation)
            mplot.plot(x_Validation,y_prediction)
            mplot.show()
    count=count+1
    RootMeanSquareError_Training_Average=np.append(RootMeanSquareError_Training_Average,np.mean(RootMeanSquareError_Training))
    RootMeanSquareError_Validation_Average=np.append(RootMeanSquareError_Validation_Average,np.mean(RootMeanSquareError_Validation))
print np.argmin(RootMeanSquareError_Training_Average)
print np.argmin(RootMeanSquareError_Validation_Average)    
sigma2=np.logspace(-4,-2,100)
print sigma1[99],sigma1[85]
RootMeanSquareError_Training_Average=np.array([])
RootMeanSquareError_Validation_Average=np.array([])
count=0
for i in sigma2:
    RootMeanSquareError_Training=np.array([])
    RootMeanSquareError_Validation=np.array([])       
    for x_Training, y_Training, x_Validation, y_Validation in KFold(Normalizedx, Normalizedy):    
        K2=np.array([])  
        for j in x_Training:
            for k in x_Training:
                kernel2=math.exp(-((j-k)*(j-k))/(i))
                K2=np.append(K2,kernel2)
        GramMatrix=np.reshape(K2,(x_Training.size,x_Training.size)) 
        I=np.identity(x_Training.size)
        C=np.add(GramMatrix,np.dot((1/Beta),I))     
        a=np.dot(np.linalg.inv(C),y_Training)
        y_prediction=np.array([])
        for m in range(0,x_Training.size):
            summation=0
            for n in range(0,x_Training.size):
                summation=summation+(a[n]*GramMatrix[m][n])
            y_prediction=np.append(y_prediction,summation)
        RootMeanSquareError_Training=np.append(RootMeanSquareError_Training,RMSE(y_prediction,y_Training))    
        if count==99:
            print y_prediction.shape      
            print y_prediction
            mplot.plot(x_Training,y_Training)
            mplot.plot(x_Training,y_prediction)
            mplot.show()  
        K2=np.array([])  
        for j in x_Validation:
            for k in x_Validation:
                kernel2=math.exp(-((j-k)*(j-k))/(i))
                K2=np.append(K2,kernel2)
        GramMatrix=np.reshape(K2,(x_Validation.size,x_Validation.size)) 
        I=np.identity(x_Validation.size)
        C=np.add(GramMatrix,np.dot((1/Beta),I))     
        a=np.dot(np.linalg.inv(C),y_Validation)
        y_prediction=np.array([])
        for m in range(0,x_Validation.size):
            summation=0
            for n in range(0,x_Validation.size):
                summation=summation+(a[n]*GramMatrix[m][n])
            y_prediction=np.append(y_prediction,summation)
        RootMeanSquareError_Validation=np.append(RootMeanSquareError_Validation,RMSE(y_prediction,y_Validation))    
        if count==85:
            print y_prediction.shape      
            print y_prediction
            mplot.plot(x_Validation,y_Validation)
            mplot.plot(x_Validation,y_prediction)
            mplot.show()  
    count=count+1
    RootMeanSquareError_Training_Average=np.append(RootMeanSquareError_Training_Average,np.mean(RootMeanSquareError_Training))
    RootMeanSquareError_Validation_Average=np.append(RootMeanSquareError_Validation_Average,np.mean(RootMeanSquareError_Validation))
print np.argmin(RootMeanSquareError_Training_Average)
print np.argmin(RootMeanSquareError_Validation_Average)