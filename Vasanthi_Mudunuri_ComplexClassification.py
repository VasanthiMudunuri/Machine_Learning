import numpy as np
import matplotlib.pyplot as mplot
from scipy.spatial import KDTree

mean = [2,2]
covariance = [[2,0],[0,2]]
x1,y1 = np.random.multivariate_normal(mean,covariance,1000).T
Distribution1=np.column_stack((x1,y1))
mean = [4,4]
covariance = [[2,0],[0,2]]
x2,y2 = np.random.multivariate_normal(mean,covariance,1000).T
Distribution2=np.column_stack((x2,y2))
mean = [6,6]
covariance = [[2,0],[0,2]]
x3,y3 = np.random.multivariate_normal(mean,covariance,1000).T
Distribution3=np.column_stack((x3,y3))
mean = [8,8]
covariance = [[2,0],[0,2]]
x4,y4 = np.random.multivariate_normal(mean,covariance,1000).T
Distribution4=np.column_stack((x4,y4))
mean = [10,10]
covariance = [[2,0],[0,2]]
x5,y5 = np.random.multivariate_normal(mean,covariance,1000).T
Distribution5=np.column_stack((x5,y5))
mean = [12,12]
covariance = [[2,0],[0,2]]
x6,y6 = np.random.multivariate_normal(mean,covariance,1000).T
Distribution6=np.column_stack((x6,y6))
mean = [14,14]
covariance = [[2,0],[0,2]]
x7,y7 = np.random.multivariate_normal(mean,covariance,1000).T
Distribution7=np.column_stack((x7,y7))
mean = [16,16]
covariance = [[2,0],[0,2]]
x8,y8 = np.random.multivariate_normal(mean,covariance,1000).T
Distribution8=np.column_stack((x8,y8))
mean = [18,18]
covariance = [[2,0],[0,2]]
x9,y9 = np.random.multivariate_normal(mean,covariance,1000).T
Distribution9=np.column_stack((x9,y9))
mean = [20,20]
covariance = [[2,0],[0,2]]
x10,y10 = np.random.multivariate_normal(mean,covariance,1000).T
Distribution10=np.column_stack((x10,y10))
X=np.concatenate((Distribution1,Distribution2,Distribution3,Distribution4,Distribution5,
                  Distribution6,Distribution7,Distribution8,Distribution9,Distribution10),axis =0)
mplot.plot(x1,y1,'x')
mplot.plot(x2,y2,'x')
mplot.plot(x3,y3,'x')
mplot.plot(x4,y4,'x')
mplot.plot(x5,y5,'x')
mplot.plot(x6,y6,'x')
mplot.plot(x7,y7,'x')
mplot.plot(x8,y8,'x')
mplot.plot(x9,y9,'x')
mplot.plot(x10,y10,'x')
mplot.show()
Column1=np.zeros((5000,1))
Column2=np.ones((5000,1))
Y=np.concatenate((Column1,Column2),axis=0)
#Mask=np.random.choice(a=[False, True], size=(10000,1))
mask=np.random.rand(10000)<0.8
training_X=X[mask]
training_Y=Y[mask]
mask=np.logical_not(mask)
test_X=X[mask]
test_Y=Y[mask]
Training_Data=(training_X.transpose()).dot(training_X)
Inverse=(np.linalg.inv(Training_Data))
Training_transpose=(Inverse).dot(training_X.transpose())
Beta=(Training_transpose).dot(training_Y)
Condition=(test_X).dot(Beta)
New_Y= np.matrix(Condition >= 0.5).astype(int)
accuracy = (New_Y == test_Y).mean()
error=(New_Y != test_Y).mean()
print('Accuracy:',accuracy)
print('Error:',error)
A=np.concatenate((Distribution1,Distribution2,Distribution3,Distribution4,Distribution5),axis =0)
B=np.array(training_X)
aset=set([tuple(p) for p in A])
bset=set([tuple(q) for q in B])
Training_class0=np.array([r for r in aset & bset])
Training_class0_x=Training_class0[:,0]
Training_class0_y=Training_class0[:,1]
C=np.concatenate((Distribution6,Distribution7,Distribution8,Distribution9,Distribution10),axis =0)
cset=set([tuple(p) for p in C])
Training_class1=np.array([s for s in bset & cset])
Training_class1_x=Training_class1[:,0]
Training_class1_y=Training_class1[:,1]
test_1=np.nonzero(test_Y)[0]
find_1=np.squeeze(np.asarray(np.nonzero(New_Y)[0]))
test_0=np.where(test_Y == 0)[0]
find_0=np.squeeze(np.asarray(np.where(New_Y == 0)[0]))
Correct_class1=np.array(list(set(test_1).intersection(set(find_1)))).astype(int)
Correct_class0=np.array(list(set(test_0).intersection(set(find_0))))
Elements_Correctclass0=training_X[Correct_class0]
Elements_Correctclass0_x=Elements_Correctclass0[:,0]
Elements_Correctclass0_y=Elements_Correctclass0[:,1]
Elements_Correctclass1=training_X[Correct_class1]
Elements_Correctclass1_x=Elements_Correctclass1[:,0]
Elements_Correctclass1_y=Elements_Correctclass1[:,1]
Elements_Incorrectclass0=training_X[Correct_class0.size-Correct_class0]
Elements_Incorrectclass0_x=Elements_Correctclass0[:,0]
Elements_Incorrectclass0_y=Elements_Correctclass0[:,1]
Elements_Incorrectclass1=training_X[Correct_class1.size-Correct_class1]
Elements_Incorrectclass1_x=Elements_Correctclass1[:,0]
Elements_Incorrectclass1_y=Elements_Correctclass1[:,1]
mplot.plot(Training_class0_x,Training_class0_y,'x',color='b')
mplot.plot(Training_class1_x,Training_class1_y,'x',color='m')
mplot.plot(Elements_Correctclass0_x,Elements_Correctclass0_y,'x',color='r')
mplot.plot(Elements_Correctclass1_x,Elements_Correctclass1_y,'x',color='g')
mplot.plot(Elements_Incorrectclass0_x,Elements_Incorrectclass0_y,'x',color='c')
mplot.plot(Elements_Incorrectclass1_x,Elements_Incorrectclass1_y,'x',color='y')
mplot.show()
KDT=KDTree(training_X)
Nearest_KDT=test_X
Result=KDT.query(Nearest_KDT)
KDTree_Y=np.matrix(Result).astype(int).transpose()
KDTree_Y_class0=np.matrix(Result[1] <=5000).astype(int).transpose()
KDTree_Y_class1=np.matrix(Result[1] >5000).astype(int).transpose()
accuracy = (KDTree_Y == test_Y).mean()
error=(KDTree_Y != test_Y).mean()
print('Accuracy_KDT',accuracy)
print('Error_KDT',error)
Correct_class0_kdt=np.where(np.equal(test_Y, KDTree_Y_class0))[0].tolist()
Elements_Correctclass0_kdt=training_X[Correct_class0_kdt]
Elements_Correctclass0_kdt_x=Elements_Correctclass0[:,0]
Elements_Correctclass0_kdt_y=Elements_Correctclass0[:,1]
Correct_class1_kdt=np.where(np.equal(test_Y, KDTree_Y_class1))[0].tolist()
Elements_Correctclass1_kdt=training_X[Correct_class1_kdt]
Elements_Correctclass1_kdt_x=Elements_Correctclass1[:,0]
Elements_Correctclass1_kdt_y=Elements_Correctclass1[:,1]
Incorrect_class0_kdt=np.where(np.not_equal(test_Y, KDTree_Y_class0))[0].tolist()
Elements_Incorrectclass0_kdt=training_X[Incorrect_class0_kdt]
Elements_Incorrectclass0_kdt_x=Elements_Correctclass0[:,0]
Elements_Incorrectclass0_kdt_y=Elements_Correctclass0[:,1]
Incorrect_class1_kdt=np.where(np.not_equal(test_Y, KDTree_Y_class0))[0].tolist()
Elements_Incorrectclass1_kdt=training_X[Incorrect_class1_kdt]
Elements_Incorrectclass1_kdt_x=Elements_Correctclass1[:,0]
Elements_Incorrectclass1_kdt_y=Elements_Correctclass1[:,1]
mplot.plot(Training_class0_x,Training_class0_y,'x',color='b')
mplot.plot(Training_class1_x,Training_class1_y,'x',color='m')
mplot.plot(Elements_Correctclass0_kdt_x,Elements_Correctclass0_kdt_y,'x',color='r')
mplot.plot(Elements_Correctclass1_kdt_x,Elements_Correctclass1_kdt_y,'x',color='g')
mplot.plot(Elements_Incorrectclass0_kdt_x,Elements_Incorrectclass0_kdt_y,'x',color='c')
mplot.plot(Elements_Incorrectclass1_kdt_x,Elements_Incorrectclass1_kdt_y,'x',color='y')
mplot.show()