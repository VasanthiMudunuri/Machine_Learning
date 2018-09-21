import numpy as np
import matplotlib.pyplot as mplot
from scipy.spatial import KDTree

mean = [4,4]
covariance = [[2,0],[0,2]]
x,y = np.random.multivariate_normal(mean,covariance,5000).T
Distribution1=np.column_stack((x,y))
mean = [2,2]
covariance = [[2,0],[0,2]]
a,b = np.random.multivariate_normal(mean,covariance,5000).T
Distribution2=np.column_stack((a,b))
X=np.concatenate((Distribution1,Distribution2),axis =0)
mplot.plot(x,y,'x',color='b')
mplot.plot(a,b,'x',color='m')
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
A=np.array(Distribution1)
B=np.array(training_X)
aset=set([tuple(p) for p in A])
bset=set([tuple(q) for q in B])
Training_class0=np.array([r for r in aset & bset])
Training_class0_x=Training_class0[:,0]
Training_class0_y=Training_class0[:,1]
C=np.array(Distribution2)
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
mplot.plot(Elements_Incorrectclass1_kdt_y,Elements_Incorrectclass1_kdt_y,'x',color='y')
mplot.show()