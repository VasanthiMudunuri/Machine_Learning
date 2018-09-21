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
Polynomial1=np.reshape((np.array(x1)),(46,1))
Fi1=Polynomial1     
Fi1Transpose=np.transpose(Fi1)
a1=np.dot(Fi1Transpose,Fi1) 
b1=np.dot(Fi1Transpose,y1) 
w1=np.linalg.solve(a1,b1)
print w1
Polynomial2=np.reshape((np.array(x1*x1)),(46,1))
Fi2=Polynomial2     
Fi2Transpose=np.transpose(Fi2)
a2=np.dot(Fi2Transpose,Fi2) 
b2=np.dot(Fi2Transpose,y1) 
w2=np.linalg.solve(a2,b2)
print w2
Polynomial3=np.reshape((np.array(x1*x1*x1)),(46,1))
Fi3=Polynomial3     
Fi3Transpose=np.transpose(Fi3)
a3=np.dot(Fi3Transpose,Fi3) 
b3=np.dot(Fi3Transpose,y1) 
w3=np.linalg.solve(a3,b3)
print w3
Polynomial4=np.reshape((np.array(x1*x1*x1*x1)),(46,1))
Fi4=Polynomial4     
Fi4Transpose=np.transpose(Fi4)
a4=np.dot(Fi4Transpose,Fi4) 
b4=np.dot(Fi4Transpose,y1) 
w4=np.linalg.solve(a4,b4)
print w4
Polynomial5=np.reshape((np.array(x1*x1*x1*x1*x1)),(46,1))
Fi5=Polynomial5     
Fi5Transpose=np.transpose(Fi5)
a5=np.dot(Fi5Transpose,Fi5) 
b5=np.dot(Fi5Transpose,y1) 
w5=np.linalg.solve(a5,b5)
print w5
Polynomial6=np.reshape((np.array(x1*x1*x1*x1*x1*x1)),(46,1))
Fi6=Polynomial6     
Fi6Transpose=np.transpose(Fi6)
a6=np.dot(Fi6Transpose,Fi6) 
b6=np.dot(Fi6Transpose,y1) 
w6=np.linalg.solve(a6,b6)
print w6
Polynomial7=np.reshape((np.array(x1*x1*x1*x1*x1*x1*x1)),(46,1))
Fi7=Polynomial7     
Fi7Transpose=np.transpose(Fi7)
a7=np.dot(Fi7Transpose,Fi7) 
b7=np.dot(Fi7Transpose,y1) 
w7=np.linalg.solve(a7,b7)
print w7
Polynomial8=np.reshape((np.array(x1*x1*x1*x1*x1*x1*x1*x1)),(46,1))
Fi8=Polynomial8    
Fi8Transpose=np.transpose(Fi8)
a8=np.dot(Fi8Transpose,Fi8) 
b8=np.dot(Fi8Transpose,y1) 
w8=np.linalg.solve(a8,b8)
print w8
Polynomial9=np.reshape((np.array(x1*x1*x1*x1*x1*x1*x1*x1*x1)),(46,1))
Fi9=Polynomial9    
Fi9Transpose=np.transpose(Fi9)
a9=np.dot(Fi9Transpose,Fi9) 
b9=np.dot(Fi9Transpose,y1) 
w9=np.linalg.solve(a9,b9)
print w9
Polynomial10=np.reshape((np.array(x1*x1*x1*x1*x1*x1*x1*x1*x1*x1)),(46,1))
Fi10=Polynomial10   
Fi10Transpose=np.transpose(Fi10)
a10=np.dot(Fi10Transpose,Fi10) 
b10=np.dot(Fi10Transpose,y1) 
w10=np.linalg.solve(a10,b10)
print w10
Polynomial11=np.reshape((np.array(x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1)),(46,1))
Fi11=Polynomial11   
Fi11Transpose=np.transpose(Fi11)
a11=np.dot(Fi11Transpose,Fi11) 
b11=np.dot(Fi11Transpose,y1) 
w11=np.linalg.solve(a11,b11)
print w11
Polynomial12=np.reshape((np.array(x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1)),(46,1))
Fi12=Polynomial12   
Fi12Transpose=np.transpose(Fi12)
a12=np.dot(Fi12Transpose,Fi12) 
b12=np.dot(Fi12Transpose,y1) 
w12=np.linalg.solve(a12,b12)
print w12
Polynomial13=np.reshape((np.array(x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1)),(46,1))
Fi13=Polynomial13   
Fi13Transpose=np.transpose(Fi13)
a13=np.dot(Fi13Transpose,Fi13) 
b13=np.dot(Fi13Transpose,y1) 
w13=np.linalg.solve(a13,b13)
print w13
Polynomial14=np.reshape((np.array(x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1)),(46,1))
Fi14=Polynomial14   
Fi14Transpose=np.transpose(Fi14)
a14=np.dot(Fi14Transpose,Fi14) 
b14=np.dot(Fi14Transpose,y1) 
w14=np.linalg.solve(a14,b14)
print w14
Polynomial15=np.reshape((np.array(x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1)),(46,1))
Fi15=Polynomial15   
Fi15Transpose=np.transpose(Fi15)
a15=np.dot(Fi15Transpose,Fi15) 
b15=np.dot(Fi15Transpose,y1) 
w15=np.linalg.solve(a15,b15)
print w15
Polynomial16=np.reshape((np.array(x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1)),(46,1))
Fi16=Polynomial16  
Fi16Transpose=np.transpose(Fi16)
a16=np.dot(Fi16Transpose,Fi16) 
b16=np.dot(Fi16Transpose,y1) 
w16=np.linalg.solve(a16,b16)
print w16
Polynomial17=np.reshape((np.array(x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1)),(46,1))
Fi17=Polynomial17   
Fi17Transpose=np.transpose(Fi17)
a17=np.dot(Fi17Transpose,Fi17) 
b17=np.dot(Fi17Transpose,y1) 
w17=np.linalg.solve(a17,b17)
print w17
Polynomial18=np.reshape((np.array(x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1)),(46,1))
Fi18=Polynomial18   
Fi18Transpose=np.transpose(Fi18)
a18=np.dot(Fi18Transpose,Fi18) 
b18=np.dot(Fi18Transpose,y1) 
w18=np.linalg.solve(a18,b18)
print w18
Polynomial19=np.reshape((np.array(x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1)),(46,1))
Fi19=Polynomial19   
Fi19Transpose=np.transpose(Fi19)
a19=np.dot(Fi19Transpose,Fi19) 
b19=np.dot(Fi19Transpose,y1) 
w19=np.linalg.solve(a19,b19)
print w19
Polynomial20=np.reshape((np.array(x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1*x1)),(46,1))
Fi20=Polynomial20   
Fi20Transpose=np.transpose(Fi20)
a20=np.dot(Fi20Transpose,Fi20) 
b20=np.dot(Fi20Transpose,y1) 
w20=np.linalg.solve(a20,b20)
print w20

Polynomial=np.concatenate((Polynomial1,Polynomial2,Polynomial3,Polynomial4,Polynomial5,
                           Polynomial6,Polynomial7,Polynomial8,Polynomial9,Polynomial10,
                           Polynomial11,Polynomial12,Polynomial13,Polynomial14,Polynomial15,
                           Polynomial16,Polynomial17,Polynomial18,Polynomial19,Polynomial20),axis=1)                 
w=np.concatenate((w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16,w17,w18,w19,w20),axis=0)   
#print w
Fi=np.reshape(Polynomial,(46,20))
weight=np.reshape((np.repeat(np.reshape(w,(20,1)),20)),(20,20))
#print weight
FiTranspose=np.transpose(Fi)
equation=np.dot(FiTranspose,Fi)
a=FiTranspose
b=np.dot(equation,weight)
prediction=np.linalg.lstsq(a,b)[0]
#print prediction
actual=np.reshape((np.repeat(np.reshape(y1,(46,1)),20)),(46,20))
#print actual
diff=np.subtract(actual,prediction)
squareofdiff=np.square(diff)
sumofsquareofdiff=np.sum(squareofdiff,axis=0)
RMS=np.sqrt(np.divide(sumofsquareofdiff,46))
mplot.plot(RMS,'-')
mplot.show()
RMSmin=np.argmin(RMS)
print RMSmin
#x=np.linspace(0,60,1,endpoint=True)
#mplot.plot(x)
mplot.plot(x1,y1,'o')
mplot.plot(prediction[:,0],'-')
mplot.show()

Tpolynomial1=np.reshape((np.array(x2)),(47,1))
Tpolynomial2=np.reshape((np.array(x2*x2)),(47,1))
Tpolynomial3=np.reshape((np.array(x2*x2*x2)),(47,1))
Tpolynomial4=np.reshape((np.array(x2*x2*x2*x2)),(47,1))
Tpolynomial5=np.reshape((np.array(x2*x2*x2*x2*x2)),(47,1))
Tpolynomial6=np.reshape((np.array(x2*x2*x2*x2*x2*x2)),(47,1))
Tpolynomial7=np.reshape((np.array(x2*x2*x2*x2*x2*x2*x2)),(47,1))
Tpolynomial8=np.reshape((np.array(x2*x2*x2*x2*x2*x2*x2*x2)),(47,1))
Tpolynomial9=np.reshape((np.array(x2*x2*x2*x2*x2*x2*x2*x2*x2)),(47,1))
Tpolynomial10=np.reshape((np.array(x2*x2*x2*x2*x2*x2*x2*x2*x2*x2)),(47,1))
Tpolynomial11=np.reshape((np.array(x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2)),(47,1))
Tpolynomial12=np.reshape((np.array(x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2)),(47,1))
Tpolynomial13=np.reshape((np.array(x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2)),(47,1))
Tpolynomial14=np.reshape((np.array(x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2)),(47,1))
Tpolynomial15=np.reshape((np.array(x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2)),(47,1))
Tpolynomial16=np.reshape((np.array(x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2)),(47,1))
Tpolynomial17=np.reshape((np.array(x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2)),(47,1))
Tpolynomial18=np.reshape((np.array(x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2)),(47,1))
Tpolynomial19=np.reshape((np.array(x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2)),(47,1))
Tpolynomial20=np.reshape((np.array(x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2*x2)),(47,1))

TPolynomial=np.concatenate((Tpolynomial1,Tpolynomial2,Tpolynomial3,Tpolynomial4,Tpolynomial5,
                           Tpolynomial6,Tpolynomial7,Tpolynomial8,Tpolynomial9,Tpolynomial10,
                           Tpolynomial11,Tpolynomial12,Tpolynomial13,Tpolynomial14,Tpolynomial15,
                           Tpolynomial16,Tpolynomial17,Tpolynomial18,Tpolynomial19,Tpolynomial20),axis=1)                      
TFi=np.reshape(TPolynomial,(47,20))   
TFiTranspose=np.transpose(TFi)
Ta=np.dot(TFiTranspose,TFi) 
b=np.dot(Ta,weight)
#print b.shape
a=TFiTranspose
#print a.shape
prediction=np.linalg.lstsq(a,b)[0]
print prediction
actual=np.reshape((np.repeat(np.reshape(y2,(47,1)),20)),(47,20))
result=np.subtract(actual,prediction)
squareofdiff=np.square(result)
sumofsquareofdiff=np.sum(squareofdiff,axis=0)
print sumofsquareofdiff.shape
RMS=np.sqrt(np.divide(sumofsquareofdiff,47))
mplot.plot(RMS,'-')
mplot.show()
RMSmin=np.argmin(RMS)
print RMSmin
mplot.plot(x2,y2,'o')
mplot.plot(prediction[:,0],'-')
mplot.show()