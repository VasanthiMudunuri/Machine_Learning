import numpy as np
import matplotlib.pyplot as mplot

Data=np.loadtxt('C:\crash.txt')

TrainingData=Data[1::2]
TestData=Data[::2]
x1=np.array(TrainingData[:,0])
y1=np.array(TrainingData[:,1])
x2=np.array(TestData[:,0])
y2=np.array(TestData[:,1])
data=np.reshape((np.array(x1)),(46,1))
beta=0.0025
alpha=np.logspace(-8,0,100)

Identity=np.identity(40,dtype=int)
Fi1=np.exp(-1*(np.divide(np.square(np.subtract(data,1.5)),2*((1.5)**2))))  
Fi2=np.exp(-1*(np.divide(np.square(np.subtract(data,3)),2*((1.5)**2))))  
Fi3=np.exp(-1*(np.divide(np.square(np.subtract(data,4.5)),2*((1.5)**2))))  
Fi4=np.exp(-1*(np.divide(np.square(np.subtract(data,6)),2*((1.5)**2))))  
Fi5=np.exp(-1*(np.divide(np.square(np.subtract(data,7.5)),2*((1.5)**2))))  
Fi6=np.exp(-1*(np.divide(np.square(np.subtract(data,9)),2*((1.5)**2))))  
Fi7=np.exp(-1*(np.divide(np.square(np.subtract(data,10.5)),2*((1.5)**2))))  
Fi8=np.exp(-1*(np.divide(np.square(np.subtract(data,12)),2*((1.5)**2))))  
Fi9=np.exp(-1*(np.divide(np.square(np.subtract(data,13.5)),2*((1.5)**2))))  
Fi10=np.exp(-1*(np.divide(np.square(np.subtract(data,15)),2*((1.5)**2))))  
Fi11=np.exp(-1*(np.divide(np.square(np.subtract(data,16.5)),2*((1.5)**2))))  
Fi12=np.exp(-1*(np.divide(np.square(np.subtract(data,18)),2*((1.5)**2))))  
Fi13=np.exp(-1*(np.divide(np.square(np.subtract(data,19.5)),2*((1.5)**2))))  
Fi14=np.exp(-1*(np.divide(np.square(np.subtract(data,21)),2*((1.5)**2))))  
Fi15=np.exp(-1*(np.divide(np.square(np.subtract(data,22.5)),2*((1.5)**2))))  
Fi16=np.exp(-1*(np.divide(np.square(np.subtract(data,24)),2*((1.5)**2))))  
Fi17=np.exp(-1*(np.divide(np.square(np.subtract(data,25.5)),2*((1.5)**2))))  
Fi18=np.exp(-1*(np.divide(np.square(np.subtract(data,27)),2*((1.5)**2))))  
Fi19=np.exp(-1*(np.divide(np.square(np.subtract(data,28.5)),2*((1.5)**2))))  
Fi20=np.exp(-1*(np.divide(np.square(np.subtract(data,30)),2*((1.5)**2))))  
Fi21=np.exp(-1*(np.divide(np.square(np.subtract(data,31.5)),2*((1.5)**2))))  
Fi22=np.exp(-1*(np.divide(np.square(np.subtract(data,33)),2*((1.5)**2))))  
Fi23=np.exp(-1*(np.divide(np.square(np.subtract(data,34.5)),2*((1.5)**2))))  
Fi24=np.exp(-1*(np.divide(np.square(np.subtract(data,36)),2*((1.5)**2))))  
Fi25=np.exp(-1*(np.divide(np.square(np.subtract(data,37.5)),2*((1.5)**2))))  
Fi26=np.exp(-1*(np.divide(np.square(np.subtract(data,39)),2*((1.5)**2))))  
Fi27=np.exp(-1*(np.divide(np.square(np.subtract(data,40.5)),2*((1.5)**2))))  
Fi28=np.exp(-1*(np.divide(np.square(np.subtract(data,42)),2*((1.5)**2))))  
Fi29=np.exp(-1*(np.divide(np.square(np.subtract(data,43.5)),2*((1.5)**2))))  
Fi30=np.exp(-1*(np.divide(np.square(np.subtract(data,45)),2*((1.5)**2))))  
Fi31=np.exp(-1*(np.divide(np.square(np.subtract(data,46.5)),2*((1.5)**2))))  
Fi32=np.exp(-1*(np.divide(np.square(np.subtract(data,48)),2*((1.5)**2))))  
Fi33=np.exp(-1*(np.divide(np.square(np.subtract(data,49.5)),2*((1.5)**2))))  
Fi34=np.exp(-1*(np.divide(np.square(np.subtract(data,51)),2*((1.5)**2))))  
Fi35=np.exp(-1*(np.divide(np.square(np.subtract(data,52.5)),2*((1.5)**2))))  
Fi36=np.exp(-1*(np.divide(np.square(np.subtract(data,54)),2*((1.5)**2))))  
Fi37=np.exp(-1*(np.divide(np.square(np.subtract(data,55.5)),2*((1.5)**2))))  
Fi38=np.exp(-1*(np.divide(np.square(np.subtract(data,57)),2*((1.5)**2))))  
Fi39=np.exp(-1*(np.divide(np.square(np.subtract(data,58.5)),2*((1.5)**2))))  
Fi40=np.exp(-1*(np.divide(np.square(np.subtract(data,60)),2*((1.5)**2))))  

Fi=np.concatenate((Fi1,Fi2,Fi3,Fi4,Fi5,Fi6,Fi7,Fi8,Fi9,Fi10,Fi11,Fi12,Fi13,Fi14,Fi15,
                   Fi16,Fi17,Fi18,Fi19,Fi20,Fi21,Fi22,Fi23,Fi24,Fi25,Fi26,Fi27,Fi28,Fi29,Fi30,
                   Fi31,Fi32,Fi33,Fi34,Fi35,Fi36,Fi37,Fi38,Fi39,Fi40),axis=1)
FiTranspose=np.transpose(Fi)
weight=np.array([])
for i in alpha:
    a1=np.dot(FiTranspose,Fi) 
    a2=Identity*(i/beta)
    a=a1+a2
    b1=np.dot(FiTranspose,y1) 
    w1=np.linalg.solve(a,b1)
    #print w1
    weight=np.append(weight,w1)
    print weight
weight1=weight[::40]
print '*********************',weight1
weight2=weight[1::40]
weight3=weight[2::40]
weight4=weight[3::40]
weight5=weight[4::40]
weight6=weight[5::40]
weight7=weight[6::40]
weight8=weight[7::40]
weight9=weight[8::40]
weight10=weight[9::40]
weight11=weight[10::40]
weight12=weight[11::40]
weight13=weight[12::40]
weight14=weight[13::40]
weight15=weight[14::40]
weight16=weight[15::40]
weight17=weight[16::40]
weight18=weight[17::40]
weight19=weight[18::40]
weight20=weight[19::40]
weight21=weight[20::40]
weight22=weight[21::40]
weight23=weight[22::40]
weight24=weight[23::40]
weight25=weight[24::40]
weight26=weight[25::40]
weight27=weight[26::40]
weight28=weight[27::40]
weight29=weight[28::40]
weight30=weight[29::40]
weight31=weight[30::40]
weight32=weight[31::40]
weight33=weight[32::40]
weight34=weight[33::40]
weight35=weight[34::40]
weight36=weight[35::40]
weight37=weight[36::40]
weight38=weight[37::40]
weight39=weight[38::40]
weight40=weight[39::40]
print '*********************',weight40
weight=np.reshape((np.concatenate((weight1,weight2,weight3,weight4,weight5,weight6,
                                   weight7,weight8,weight9,weight10,weight11,weight12,
                                   weight13,weight14,weight15,weight16,weight17,weight18,
                                   weight19,weight20,weight21,weight22,weight23,weight24,
                                   weight25,weight26,weight27,weight28,weight29,weight30,
                                   weight31,weight32,weight33,weight34,weight35,weight36,
                                   weight37,weight38,weight39,weight40),axis=0)),(40,100))    
print weight
Tdata=np.reshape((np.array(x2)),(47,1))
beta=0.0025
alpha=np.logspace(-8,0,100)

TFi1=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,1.5)),2*((1.5)**2))))  
TFi2=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,3)),2*((1.5)**2))))  
TFi3=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,4.5)),2*((1.5)**2))))  
TFi4=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,6)),2*((1.5)**2))))  
TFi5=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,7.5)),2*((1.5)**2))))  
TFi6=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,9)),2*((1.5)**2))))  
TFi7=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,10.5)),2*((1.5)**2))))  
TFi8=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,12)),2*((1.5)**2))))  
TFi9=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,13.5)),2*((1.5)**2))))  
TFi10=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,15)),2*((1.5)**2))))  
TFi11=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,16.5)),2*((1.5)**2))))  
TFi12=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,18)),2*((1.5)**2))))  
TFi13=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,19.5)),2*((1.5)**2))))  
TFi14=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,21)),2*((1.5)**2))))  
TFi15=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,22.5)),2*((1.5)**2))))  
TFi16=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,24)),2*((1.5)**2))))  
TFi17=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,25.5)),2*((1.5)**2))))  
TFi18=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,27)),2*((1.5)**2))))  
TFi19=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,28.5)),2*((1.5)**2))))  
TFi20=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,30)),2*((1.5)**2))))  
TFi21=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,31.5)),2*((1.5)**2))))  
TFi22=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,33)),2*((1.5)**2))))  
TFi23=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,34.5)),2*((1.5)**2))))  
TFi24=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,36)),2*((1.5)**2))))  
TFi25=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,37.5)),2*((1.5)**2))))  
TFi26=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,39)),2*((1.5)**2))))  
TFi27=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,40.5)),2*((1.5)**2))))  
TFi28=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,42)),2*((1.5)**2))))  
TFi29=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,43.5)),2*((1.5)**2))))  
TFi30=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,45)),2*((1.5)**2))))  
TFi31=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,46.5)),2*((1.5)**2))))  
TFi32=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,48)),2*((1.5)**2))))  
TFi33=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,49.5)),2*((1.5)**2))))  
TFi34=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,51)),2*((1.5)**2))))  
TFi35=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,52.5)),2*((1.5)**2))))  
TFi36=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,54)),2*((1.5)**2))))  
TFi37=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,55.5)),2*((1.5)**2))))  
TFi38=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,57)),2*((1.5)**2))))  
TFi39=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,58.5)),2*((1.5)**2))))  
TFi40=np.exp(-1*(np.divide(np.square(np.subtract(Tdata,60)),2*((1.5)**2))))  


TFi=np.concatenate((TFi1,TFi2,TFi3,TFi4,TFi5,TFi6,TFi7,TFi8,TFi9,TFi10,TFi11,TFi12,TFi13,TFi14,TFi15,
                   TFi16,TFi17,TFi18,TFi19,TFi20,TFi21,TFi22,TFi23,TFi24,TFi25,TFi26,TFi27,TFi28,TFi29,TFi30,
                   TFi31,TFi32,TFi33,TFi34,TFi35,TFi36,TFi37,TFi38,TFi39,TFi40),axis=1)
TFiTranspose=np.transpose(TFi)
RMS=np.array([])
j=0
for i in alpha:
        a1=np.dot(TFiTranspose,TFi)
        a2=Identity*(i/beta)
        a=TFiTranspose
        #print weight[:,j]
        b=np.dot((a1+a2),np.reshape(weight[:,j],(40,1)))
        prediction=np.linalg.lstsq(a,b)[0]
        actual=np.reshape((np.repeat(np.reshape(y2,(47,1)),40)),(47,40))    
        diff=np.subtract(actual,prediction)
        squareofdiff=np.square(diff)
        sumofsquareofdiff=np.sum(squareofdiff,axis=0)
        rms=np.sqrt(np.divide(sumofsquareofdiff,47))
        #print rms.shape
        RMS=np.append(RMS,rms)
        #print rms
        j=j+1
RMS1=RMS[0::40]
RMS2=RMS[1::40]
RMS3=RMS[2::40]
RMS4=RMS[3::40]
RMS5=RMS[4::40]
RMS6=RMS[5::40]
RMS7=RMS[6::40]
RMS8=RMS[7::40]
RMS9=RMS[8::40]
RMS10=RMS[9::40]
RMS11=RMS[10::40]
RMS12=RMS[11::40]
RMS13=RMS[12::40]
RMS14=RMS[13::40]
RMS15=RMS[14::40]
RMS16=RMS[15::40]
RMS17=RMS[16::40]
RMS18=RMS[17::40]
RMS19=RMS[18::40]
RMS20=RMS[19::40]
RMS21=RMS[20::40]
RMS22=RMS[21::40]
RMS23=RMS[22::40]
RMS24=RMS[23::40]
RMS25=RMS[24::40]
RMS26=RMS[25::40]
RMS27=RMS[26::40]
RMS28=RMS[27::40]
RMS29=RMS[28::40]
RMS30=RMS[29::40]
RMS31=RMS[30::40]
RMS32=RMS[31::40]
RMS33=RMS[32::40]
RMS34=RMS[33::40]
RMS35=RMS[34::40]
RMS36=RMS[35::40]
RMS37=RMS[36::40]
RMS38=RMS[37::40]
RMS39=RMS[38::40]
RMS40=RMS[39::40]
RMS=np.reshape((np.concatenate((RMS1,RMS2,RMS3,RMS4,RMS5,RMS6,RMS7,RMS8,RMS9,RMS10,
                                RMS11,RMS12,RMS13,RMS14,RMS15,RMS16,RMS17,RMS18,RMS19,RMS20,
                                RMS21,RMS22,RMS23,RMS24,RMS25,RMS26,RMS27,RMS28,RMS29,RMS30,
                                RMS31,RMS32,RMS33,RMS34,RMS35,RMS36,RMS37,RMS38,RMS39,RMS40,),axis=0)),(40,100))  
#print RMS 
alphasum=np.sum(RMS,axis=0)
alphaavg=np.divide(alphasum,40)
Best=np.argmin(alphaavg)
print Best

a1=np.dot(TFiTranspose,TFi)
a2=Identity*(alpha[99]/beta)
a=TFiTranspose
b=np.dot((a1+a2),np.reshape(weight[:,99],(40,1)))
prediction=np.linalg.lstsq(a,b)[0]
mplot.plot(x2,y2,'o')
mplot.plot(prediction,'-')
mplot.show()