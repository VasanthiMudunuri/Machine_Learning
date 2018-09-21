import numpy as np
import matplotlib.pyplot as mplot

Data=np.loadtxt('C:\crash.txt')
print Data
TrainingData=Data[1::2]
TestData=Data[::2]
x1=np.array(TrainingData[:,0])
y1=np.array(TrainingData[:,1])
x2=np.array(TestData[:,0])
y2=np.array(TestData[:,1])
data=np.reshape((np.array(x1)),(46,1))
Fi1=np.exp(-1*(np.divide(np.square(np.subtract(data,12)),2*((12)**2))))    
Fi1Transpose=np.transpose(Fi1)
a1=np.dot(Fi1Transpose,Fi1) 
b1=np.dot(Fi1Transpose,y1) 
w1=np.linalg.solve(a1,b1)
print w1
Fi2=np.exp(-1*(np.divide(np.square(np.subtract(data,24)),2*((12)**2))))   
Fi2Transpose=np.transpose(Fi2)
a2=np.dot(Fi2Transpose,Fi2) 
b2=np.dot(Fi2Transpose,y1) 
w2=np.linalg.solve(a2,b2)
print w2
Fi3=np.exp(-1*(np.divide(np.square(np.subtract(data,36)),2*((12)**2))))   
Fi3Transpose=np.transpose(Fi3)
a3=np.dot(Fi3Transpose,Fi3) 
b3=np.dot(Fi3Transpose,y1) 
w3=np.linalg.solve(a3,b3)
print w3
Fi4=np.exp(-1*(np.divide(np.square(np.subtract(data,48)),2*((12)**2))))    
Fi4Transpose=np.transpose(Fi4)
a4=np.dot(Fi4Transpose,Fi4) 
b4=np.dot(Fi4Transpose,y1) 
w4=np.linalg.solve(a4,b4)
print w4
Fi5=np.exp(-1*(np.divide(np.square(np.subtract(data,60)),2*((12)**2))))    
Fi5Transpose=np.transpose(Fi5)
a5=np.dot(Fi5Transpose,Fi5) 
b5=np.dot(Fi5Transpose,y1) 
w5=np.linalg.solve(a5,b5)
print w5
                      
w=np.concatenate((w1,w2,w3,w4,w5),axis=0)   
print w
Fi=np.concatenate((Fi1,Fi2,Fi3,Fi4,Fi5),axis=1)
weight=np.reshape((np.repeat(np.reshape(w,(5,1)),5)),(5,5))
print weight
FiTranspose=np.transpose(Fi)
equation=np.dot(FiTranspose,Fi)
a=FiTranspose
b=np.dot(equation,weight)
prediction=np.linalg.lstsq(a,b)[0]
print prediction
actual=np.reshape((np.repeat(np.reshape(y1,(46,1)),5)),(46,5))
diff=np.subtract(actual,prediction)
squareofdiff=np.square(diff)
sumofsquareofdiff=np.sum(squareofdiff,axis=0)
RMS=np.sqrt(np.divide(sumofsquareofdiff,46))
print RMS
RMSmin=np.argmin(RMS)
print RMSmin
mplot.plot(x1,y1,'o')
mplot.plot(prediction[:,0],'-')
mplot.show()

Tdata=np.reshape((np.array(x2)),(47,1))
TFi1=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,12)),2*((12)**2))))    
TFi2=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,24)),2*((12)**2))))   
TFi3=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,36)),2*((12)**2))))   
TFi4=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,48)),2*((12)**2))))    
TFi5=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,60)),2*((12)**2))))    

TFi=np.concatenate((TFi1,TFi2,TFi3,TFi4,TFi5),axis=1)
Tweight=np.reshape((np.repeat(np.reshape(w,(5,1)),5)),(5,5))
print Tweight
TFiTranspose=np.transpose(TFi)
Tequation=np.dot(TFiTranspose,TFi)
Ta=TFiTranspose
Tb=np.dot(Tequation,Tweight)
Tprediction=np.linalg.lstsq(Ta,Tb)[0]
print Tprediction
Tactual=np.reshape((np.repeat(np.reshape(y2,(47,1)),5)),(47,5))
Tdiff=np.subtract(Tactual,Tprediction)
Tsquareofdiff=np.square(Tdiff)
Tsumofsquareofdiff=np.sum(Tsquareofdiff,axis=0)
TRMS=np.sqrt(np.divide(Tsumofsquareofdiff,47))
print TRMS
TRMSmin=np.argmin(TRMS)
print TRMSmin
mplot.plot(x2,y2,'o')
mplot.plot(Tprediction[:,0],'-')
mplot.show()


Fi1=np.exp(-1*(np.divide(np.square(np.subtract(data,6)),2*((6)**2))))    
Fi2=np.exp(-1*(np.divide(np.square(np.subtract(data,12)),2*((6)**2))))   
Fi3=np.exp(-1*(np.divide(np.square(np.subtract(data,18)),2*((6)**2))))   
Fi4=np.exp(-1*(np.divide(np.square(np.subtract(data,24)),2*((6)**2))))    
Fi5=np.exp(-1*(np.divide(np.square(np.subtract(data,30)),2*((6)**2))))   
Fi6=np.exp(-1*(np.divide(np.square(np.subtract(data,36)),2*((6)**2)))) 
Fi7=np.exp(-1*(np.divide(np.square(np.subtract(data,42)),2*((6)**2)))) 
Fi8=np.exp(-1*(np.divide(np.square(np.subtract(data,48)),2*((6)**2)))) 
Fi9=np.exp(-1*(np.divide(np.square(np.subtract(data,54)),2*((6)**2)))) 
Fi10=np.exp(-1*(np.divide(np.square(np.subtract(data,60)),2*((6)**2))))  
 
TRadial=np.concatenate((Fi1,Fi2,Fi3,Fi4,Fi5,Fi6,Fi7,Fi8,Fi9,Fi10),axis=1)                      
TFi=np.reshape(TRadial,(46,10))   
TFiTranspose=np.transpose(TFi)
a1=np.dot(TFiTranspose,TFi) 
b1=np.dot(TFiTranspose,y1) 
weight2=np.reshape(np.linalg.solve(a1,b1),(10,1))
print weight2
a=TFiTranspose
b=np.dot(a1,weight2)
prediction=np.linalg.lstsq(a,b)[0]
print prediction
actual=np.reshape((np.repeat(np.reshape(y1,(46,1)),10)),(46,10))
diff=np.subtract(actual,prediction)
squareofdiff=np.square(diff)
sumofsquareofdiff=np.sum(squareofdiff,axis=0)
RMS=np.sqrt(np.divide(sumofsquareofdiff,46))
print RMS
RMSmin=np.argmin(RMS)
print RMSmin
mplot.plot(x1,y1,'o')
mplot.plot(prediction[:,0],'-')
mplot.show()

Tdata=np.reshape((np.array(x2)),(47,1))
TFi1=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,6)),2*((6)**2))))    
TFi2=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,12)),2*((6)**2))))   
TFi3=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,18)),2*((6)**2))))   
TFi4=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,24)),2*((6)**2))))    
TFi5=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,30)),2*((6)**2))))    
TFi6=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,36)),2*((6)**2))))    
TFi7=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,42)),2*((6)**2))))   
TFi8=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,48)),2*((6)**2))))   
TFi9=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,54)),2*((6)**2))))    
TFi10=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,60)),2*((6)**2))))                      

TFi=np.concatenate((TFi1,TFi2,TFi3,TFi4,TFi5,TFi6,TFi7,TFi8,TFi9,TFi10),axis=1)
Tweight=np.reshape((np.repeat(np.reshape(weight2,(10,1)),10)),(10,10))
print Tweight
TFiTranspose=np.transpose(TFi)
Tequation=np.dot(TFiTranspose,TFi)
Ta=TFiTranspose
Tb=np.dot(Tequation,Tweight)
Tprediction=np.linalg.lstsq(Ta,Tb)[0]
print Tprediction
Tactual=np.reshape((np.repeat(np.reshape(y2,(47,1)),10)),(47,10))
Tdiff=np.subtract(Tactual,Tprediction)
Tsquareofdiff=np.square(Tdiff)
Tsumofsquareofdiff=np.sum(Tsquareofdiff,axis=0)
TRMS=np.sqrt(np.divide(Tsumofsquareofdiff,47))
print TRMS
TRMSmin=np.argmin(TRMS)
print TRMSmin
mplot.plot(x2,y2,'o')
mplot.plot(Tprediction[:,0],'-')
mplot.show()


Fi1=np.exp(-1*(np.divide(np.square(np.subtract(data,4)),2*((4)**2))))    
Fi2=np.exp(-1*(np.divide(np.square(np.subtract(data,8)),2*((4)**2))))   
Fi3=np.exp(-1*(np.divide(np.square(np.subtract(data,12)),2*((4)**2))))   
Fi4=np.exp(-1*(np.divide(np.square(np.subtract(data,16)),2*((4)**2))))    
Fi5=np.exp(-1*(np.divide(np.square(np.subtract(data,20)),2*((4)**2))))   
Fi6=np.exp(-1*(np.divide(np.square(np.subtract(data,24)),2*((4)**2)))) 
Fi7=np.exp(-1*(np.divide(np.square(np.subtract(data,28)),2*((4)**2)))) 
Fi8=np.exp(-1*(np.divide(np.square(np.subtract(data,32)),2*((4)**2)))) 
Fi9=np.exp(-1*(np.divide(np.square(np.subtract(data,36)),2*((4)**2)))) 
Fi10=np.exp(-1*(np.divide(np.square(np.subtract(data,40)),2*((4)**2))))  
Fi11=np.exp(-1*(np.divide(np.square(np.subtract(data,44)),2*((4)**2)))) 
Fi12=np.exp(-1*(np.divide(np.square(np.subtract(data,48)),2*((4)**2)))) 
Fi13=np.exp(-1*(np.divide(np.square(np.subtract(data,52)),2*((4)**2)))) 
Fi14=np.exp(-1*(np.divide(np.square(np.subtract(data,56)),2*((4)**2)))) 
Fi15=np.exp(-1*(np.divide(np.square(np.subtract(data,60)),2*((4)**2))))  
 
TRadial=np.concatenate((Fi1,Fi2,Fi3,Fi4,Fi5,Fi6,Fi7,Fi8,Fi9,Fi10,Fi11,Fi12,Fi13,Fi14,Fi15),axis=1)                      
TFi=np.reshape(TRadial,(46,15))   
TFiTranspose=np.transpose(TFi)
a1=np.dot(TFiTranspose,TFi) 
b1=np.dot(TFiTranspose,y1) 
weight3=np.reshape(np.linalg.solve(a1,b1),(15,1))
print weight3
a=TFiTranspose
b=np.dot(a1,weight3)
prediction=np.linalg.lstsq(a,b)[0]
print prediction
actual=np.reshape((np.repeat(np.reshape(y1,(46,1)),15)),(46,15))
diff=np.subtract(actual,prediction)
squareofdiff=np.square(diff)
sumofsquareofdiff=np.sum(squareofdiff,axis=0)
RMS=np.sqrt(np.divide(sumofsquareofdiff,46))
print RMS
RMSmin=np.argmin(RMS)
print RMSmin
mplot.plot(x1,y1,'o')
mplot.plot(prediction[:,0],'-')
mplot.show()

Tdata=np.reshape((np.array(x2)),(47,1))
TFi1=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,4)),2*((4)**2))))    
TFi2=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,8)),2*((4)**2))))   
TFi3=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,12)),2*((4)**2))))   
TFi4=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,16)),2*((4)**2))))    
TFi5=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,20)),2*((4)**2))))    
TFi6=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,24)),2*((4)**2))))    
TFi7=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,28)),2*((4)**2))))   
TFi8=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,32)),2*((4)**2))))   
TFi9=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,36)),2*((4)**2))))    
TFi10=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,40)),2*((4)**2))))                      
TFi11=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,44)),2*((4)**2))))  
TFi12=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,48)),2*((4)**2)))) 
TFi13=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,52)),2*((4)**2)))) 
TFi14=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,56)),2*((4)**2)))) 
TFi15=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,60)),2*((4)**2)))) 

TFi=np.concatenate((TFi1,TFi2,TFi3,TFi4,TFi5,TFi6,TFi7,TFi8,TFi9,TFi10,TFi11,TFi12,TFi13,TFi14,TFi15),axis=1)
Tweight=np.reshape((np.repeat(np.reshape(weight3,(15,1)),15)),(15,15))
print Tweight
TFiTranspose=np.transpose(TFi)
Tequation=np.dot(TFiTranspose,TFi)
Ta=TFiTranspose
Tb=np.dot(Tequation,Tweight)
Tprediction=np.linalg.lstsq(Ta,Tb)[0]
print Tprediction
Tactual=np.reshape((np.repeat(np.reshape(y2,(47,1)),15)),(47,15))
Tdiff=np.subtract(Tactual,Tprediction)
Tsquareofdiff=np.square(Tdiff)
Tsumofsquareofdiff=np.sum(Tsquareofdiff,axis=0)
TRMS=np.sqrt(np.divide(Tsumofsquareofdiff,47))
print TRMS
TRMSmin=np.argmin(TRMS)
print TRMSmin
mplot.plot(x2,y2,'o')
mplot.plot(Tprediction[:,0],'-')
mplot.show()


Fi1=np.exp(-1*(np.divide(np.square(np.subtract(data,3)),2*((3)**2))))    
Fi2=np.exp(-1*(np.divide(np.square(np.subtract(data,6)),2*((3)**2))))   
Fi3=np.exp(-1*(np.divide(np.square(np.subtract(data,9)),2*((3)**2))))   
Fi4=np.exp(-1*(np.divide(np.square(np.subtract(data,12)),2*((3)**2))))    
Fi5=np.exp(-1*(np.divide(np.square(np.subtract(data,15)),2*((3)**2))))   
Fi6=np.exp(-1*(np.divide(np.square(np.subtract(data,18)),2*((3)**2)))) 
Fi7=np.exp(-1*(np.divide(np.square(np.subtract(data,21)),2*((3)**2)))) 
Fi8=np.exp(-1*(np.divide(np.square(np.subtract(data,24)),2*((3)**2)))) 
Fi9=np.exp(-1*(np.divide(np.square(np.subtract(data,27)),2*((3)**2)))) 
Fi10=np.exp(-1*(np.divide(np.square(np.subtract(data,30)),2*((3)**2))))  
Fi11=np.exp(-1*(np.divide(np.square(np.subtract(data,33)),2*((3)**2)))) 
Fi12=np.exp(-1*(np.divide(np.square(np.subtract(data,36)),2*((3)**2)))) 
Fi13=np.exp(-1*(np.divide(np.square(np.subtract(data,39)),2*((3)**2)))) 
Fi14=np.exp(-1*(np.divide(np.square(np.subtract(data,42)),2*((3)**2)))) 
Fi15=np.exp(-1*(np.divide(np.square(np.subtract(data,45)),2*((3)**2))))  
Fi16=np.exp(-1*(np.divide(np.square(np.subtract(data,48)),2*((3)**2)))) 
Fi17=np.exp(-1*(np.divide(np.square(np.subtract(data,51)),2*((3)**2)))) 
Fi18=np.exp(-1*(np.divide(np.square(np.subtract(data,54)),2*((3)**2)))) 
Fi19=np.exp(-1*(np.divide(np.square(np.subtract(data,57)),2*((3)**2)))) 
Fi20=np.exp(-1*(np.divide(np.square(np.subtract(data,60)),2*((3)**2))))
 
TRadial=np.concatenate((Fi1,Fi2,Fi3,Fi4,Fi5,Fi6,Fi7,Fi8,Fi9,Fi10,
                        Fi11,Fi12,Fi13,Fi14,Fi15,Fi16,Fi17,Fi18,Fi19,Fi20),axis=1)                      
TFi=np.reshape(TRadial,(46,20))   
TFiTranspose=np.transpose(TFi)
a1=np.dot(TFiTranspose,TFi) 
b1=np.dot(TFiTranspose,y1) 
weight4=np.reshape(np.linalg.solve(a1,b1),(20,1))
print weight4
a=TFiTranspose
b=np.dot(a1,weight4)
prediction=np.linalg.lstsq(a,b)[0]
print prediction
actual=np.reshape((np.repeat(np.reshape(y1,(46,1)),20)),(46,20))
diff=np.subtract(actual,prediction)
squareofdiff=np.square(diff)
sumofsquareofdiff=np.sum(squareofdiff,axis=0)
RMS=np.sqrt(np.divide(sumofsquareofdiff,46))
print RMS
RMSmin=np.argmin(RMS)
print RMSmin
mplot.plot(x1,y1,'o')
mplot.plot(prediction[:,0],'-')
mplot.show()

Tdata=np.reshape((np.array(x2)),(47,1))
TFi1=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,3)),2*((3)**2))))    
TFi2=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,6)),2*((3)**2))))   
TFi3=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,9)),2*((3)**2))))   
TFi4=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,12)),2*((3)**2))))    
TFi5=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,15)),2*((3)**2))))    
TFi6=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,18)),2*((3)**2))))    
TFi7=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,21)),2*((3)**2))))   
TFi8=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,24)),2*((3)**2))))   
TFi9=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,27)),2*((3)**2))))    
TFi10=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,30)),2*((3)**2))))                      
TFi11=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,33)),2*((3)**2))))  
TFi12=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,36)),2*((3)**2)))) 
TFi13=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,39)),2*((3)**2)))) 
TFi14=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,42)),2*((3)**2)))) 
TFi15=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,45)),2*((3)**2)))) 
TFi16=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,48)),2*((3)**2))))  
TFi17=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,51)),2*((3)**2)))) 
TFi18=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,54)),2*((3)**2)))) 
TFi19=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,57)),2*((3)**2)))) 
TFi20=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,60)),2*((3)**2))))

TFi=np.concatenate((TFi1,TFi2,TFi3,TFi4,TFi5,TFi6,TFi7,TFi8,TFi9,TFi10,
                    TFi11,TFi12,TFi13,TFi14,TFi15,TFi16,TFi17,TFi18,TFi19,TFi20),axis=1)
Tweight=np.reshape((np.repeat(np.reshape(weight4,(20,1)),20)),(20,20))
print Tweight
TFiTranspose=np.transpose(TFi)
Tequation=np.dot(TFiTranspose,TFi)
Ta=TFiTranspose
Tb=np.dot(Tequation,Tweight)
Tprediction=np.linalg.lstsq(Ta,Tb)[0]
print Tprediction
Tactual=np.reshape((np.repeat(np.reshape(y2,(47,1)),20)),(47,20))
Tdiff=np.subtract(Tactual,Tprediction)
Tsquareofdiff=np.square(Tdiff)
Tsumofsquareofdiff=np.sum(Tsquareofdiff,axis=0)
TRMS=np.sqrt(np.divide(Tsumofsquareofdiff,47))
print TRMS
TRMSmin=np.argmin(TRMS)
print TRMSmin
mplot.plot(x2,y2,'o')
mplot.plot(Tprediction[:,0],'-')
mplot.show()



Fi1=np.exp(-1*(np.divide(np.square(np.subtract(data,2.4)),2*((2.4)**2))))    
Fi2=np.exp(-1*(np.divide(np.square(np.subtract(data,4.8)),2*((2.4)**2))))   
Fi3=np.exp(-1*(np.divide(np.square(np.subtract(data,7.2)),2*((2.4)**2))))   
Fi4=np.exp(-1*(np.divide(np.square(np.subtract(data,9.6)),2*((2.4)**2))))    
Fi5=np.exp(-1*(np.divide(np.square(np.subtract(data,12)),2*((2.4)**2))))   
Fi6=np.exp(-1*(np.divide(np.square(np.subtract(data,14.4)),2*((2.4)**2)))) 
Fi7=np.exp(-1*(np.divide(np.square(np.subtract(data,16.8)),2*((2.4)**2)))) 
Fi8=np.exp(-1*(np.divide(np.square(np.subtract(data,19.2)),2*((2.4)**2)))) 
Fi9=np.exp(-1*(np.divide(np.square(np.subtract(data,21.6)),2*((2.4)**2)))) 
Fi10=np.exp(-1*(np.divide(np.square(np.subtract(data,24)),2*((2.4)**2))))  
Fi11=np.exp(-1*(np.divide(np.square(np.subtract(data,26.4)),2*((2.4)**2)))) 
Fi12=np.exp(-1*(np.divide(np.square(np.subtract(data,28.8)),2*((2.4)**2)))) 
Fi13=np.exp(-1*(np.divide(np.square(np.subtract(data,31.2)),2*((2.4)**2)))) 
Fi14=np.exp(-1*(np.divide(np.square(np.subtract(data,33.6)),2*((2.4)**2)))) 
Fi15=np.exp(-1*(np.divide(np.square(np.subtract(data,36)),2*((2.4)**2))))  
Fi16=np.exp(-1*(np.divide(np.square(np.subtract(data,38.4)),2*((2.4)**2))))    
Fi17=np.exp(-1*(np.divide(np.square(np.subtract(data,40.8)),2*((2.4)**2))))   
Fi18=np.exp(-1*(np.divide(np.square(np.subtract(data,43.2)),2*((2.4)**2))))   
Fi19=np.exp(-1*(np.divide(np.square(np.subtract(data,45.6)),2*((2.4)**2))))
Fi20=np.exp(-1*(np.divide(np.square(np.subtract(data,48)),2*((2.4)**2))))      
Fi21=np.exp(-1*(np.divide(np.square(np.subtract(data,50.4)),2*((2.4)**2))))   
Fi22=np.exp(-1*(np.divide(np.square(np.subtract(data,52.8)),2*((2.4)**2)))) 
Fi23=np.exp(-1*(np.divide(np.square(np.subtract(data,55.2)),2*((2.4)**2)))) 
Fi24=np.exp(-1*(np.divide(np.square(np.subtract(data,57.6)),2*((2.4)**2)))) 
Fi25=np.exp(-1*(np.divide(np.square(np.subtract(data,60)),2*((2.4)**2)))) 

TRadial=np.concatenate((Fi1,Fi2,Fi3,Fi4,Fi5,Fi6,Fi7,Fi8,Fi9,Fi10,Fi11,Fi12,Fi13,Fi14,Fi15,
                        Fi16,Fi17,Fi18,Fi19,Fi20,Fi21,Fi22,Fi23,Fi24,Fi25),axis=1)                      
TFi=np.reshape(TRadial,(46,25))   
TFiTranspose=np.transpose(TFi)
a1=np.dot(TFiTranspose,TFi) 
b1=np.dot(TFiTranspose,y1) 
weight5=np.reshape(np.linalg.solve(a1,b1),(25,1))
print weight5
a=TFiTranspose
b=np.dot(a1,weight5)
prediction=np.linalg.lstsq(a,b)[0]
print prediction
actual=np.reshape((np.repeat(np.reshape(y1,(46,1)),25)),(46,25))
diff=np.subtract(actual,prediction)
squareofdiff=np.square(diff)
sumofsquareofdiff=np.sum(squareofdiff,axis=0)
RMS=np.sqrt(np.divide(sumofsquareofdiff,46))
print RMS
RMSmin=np.argmin(RMS)
print RMSmin
mplot.plot(x1,y1,'o')
mplot.plot(prediction[:,0],'-')
mplot.show()

Tdata=np.reshape((np.array(x2)),(47,1))
TFi1=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,2.4)),2*((2.4)**2))))    
TFi2=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,4.8)),2*((2.4)**2))))   
TFi3=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,7.2)),2*((2.4)**2))))   
TFi4=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,9.6)),2*((2.4)**2))))    
TFi5=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,12)),2*((2.4)**2))))   
TFi6=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,14.4)),2*((2.4)**2)))) 
TFi7=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,16.8)),2*((2.4)**2)))) 
TFi8=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,19.2)),2*((2.4)**2)))) 
TFi9=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,21.6)),2*((2.4)**2)))) 
TFi10=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,24)),2*((2.4)**2))))  
TFi11=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,26.4)),2*((2.4)**2)))) 
TFi12=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,28.8)),2*((2.4)**2)))) 
TFi13=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,31.2)),2*((2.4)**2)))) 
TFi14=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,33.6)),2*((2.4)**2)))) 
TFi15=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,36)),2*((2.4)**2))))  
TFi16=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,38.4)),2*((2.4)**2))))    
TFi17=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,40.8)),2*((2.4)**2))))   
TFi18=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,43.2)),2*((2.4)**2))))   
TFi19=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,45.6)),2*((2.4)**2))))
TFi20=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,48)),2*((2.4)**2))))      
TFi21=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,50.4)),2*((2.4)**2))))   
TFi22=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,52.8)),2*((2.4)**2)))) 
TFi23=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,55.2)),2*((2.4)**2)))) 
TFi24=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,57.6)),2*((2.4)**2)))) 
TFi25=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,60)),2*((2.4)**2))))

TFi=np.concatenate((TFi1,TFi2,TFi3,TFi4,TFi5,TFi6,TFi7,TFi8,TFi9,TFi10,TFi11,TFi12,TFi13,TFi14,TFi15,
                    TFi16,TFi17,TFi18,TFi19,TFi20,TFi21,TFi22,TFi23,TFi24,TFi25),axis=1)
Tweight=np.reshape((np.repeat(np.reshape(weight5,(25,1)),25)),(25,25))
print Tweight
TFiTranspose=np.transpose(TFi)
Tequation=np.dot(TFiTranspose,TFi)
Ta=TFiTranspose
Tb=np.dot(Tequation,Tweight)
Tprediction=np.linalg.lstsq(Ta,Tb)[0]
print Tprediction
Tactual=np.reshape((np.repeat(np.reshape(y2,(47,1)),25)),(47,25))
Tdiff=np.subtract(Tactual,Tprediction)
Tsquareofdiff=np.square(Tdiff)
Tsumofsquareofdiff=np.sum(Tsquareofdiff,axis=0)
TRMS=np.sqrt(np.divide(Tsumofsquareofdiff,47))
print TRMS
TRMSmin=np.argmin(TRMS)
print TRMSmin
mplot.plot(x2,y2,'o')
mplot.plot(Tprediction[:,0],'-')
mplot.show()