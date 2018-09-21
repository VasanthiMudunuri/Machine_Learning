import numpy as np
from numpy import linalg as la

def Accuracy_Rate(Actual,Prediction):
	NoOfSamples = Actual.size
	Count = 0.0
	for i in range(NoOfSamples):
            if Actual[i] == Prediction[i]:
                Count += 1.0
	return  (Count/NoOfSamples)*(100.0)
def Predict_Neighbour(Neighbours, Actual):
	Highest = [] 
	for i in Neighbours: 
          Highest.append(Actual[i])
	Highest = np.array(Highest,dtype ='int')
	return np.argmax(np.bincount(Highest))
def KNN_Point(Data, Point, K):
      X,Y= Data.shape 
      Point_Data = np.tile(Point,(X,1))
      Distance_Matrix = la.norm((Point_Data - Data),axis=1)
      Neighbour_Indexes = np.argsort(Distance_Matrix)[:K]
      return Neighbour_Indexes
def KNN(Training_Data,Training_Labels,Test_Data,K):
	Prediction = []
	for d in Test_Data:
            Neighbours = KNN_Point(Training_Data,d,K)
            Prediction.append(Predict_Neighbour(Neighbours,Training_Labels))
	return np.array(Prediction)
def main():
    training_images_file = open('C:/AI2/train-images.idx3-ubyte', 'rb')
    training_images=training_images_file.read()
    training_images_file.close()
    training_labels_file = open('C:/AI2/train-labels.idx1-ubyte', 'rb')
    training_labels=training_labels_file.read()
    training_images_file.close()
    test_images_file = open('C:/AI2/t10k-images.idx3-ubyte','rb')
    test_images=test_images_file.read()
    test_images_file.close()
    test_labels_file = open('C:/AI2/t10k-labels.idx1-ubyte','rb')
    test_labels=test_labels_file.read()
    test_labels_file.close()
    Training_images=bytearray(training_images)
    Training_labels=bytearray(training_labels)
    Test_images=bytearray(test_images)
    Test_labels=bytearray(test_labels)
    Training_Images=np.reshape(np.array(Training_images[16:]),(60000,784))
    Training_Labels=np.array(Training_labels[8:])
    Training_Images_1_list=[]
    Training_Labels_1_list=[]
    Training_Images_2_list=[]
    Training_Labels_2_list=[]
    Training_Images_7_list=[]
    Training_Labels_7_list=[]
    for i in range (60000):
        if(Training_Labels[i]==1):
            Training_Labels_1_list.append(Training_Labels[i])
            Training_Images_1_list.append(Training_Images[i]) 
        elif(Training_Labels[i]==2):
            Training_Labels_2_list.append(Training_Labels[i])
            Training_Images_2_list.append(Training_Images[i])
        elif(Training_Labels[i]==7):
            Training_Labels_7_list.append(Training_Labels[i])
            Training_Images_7_list.append(Training_Images[i]) 
    Fold_1_Images=np.concatenate((Training_Images_1_list[:40],Training_Images_2_list[:40],Training_Images_7_list[:40]),axis=0) 
    Fold_2_Images=np.concatenate((Training_Images_1_list[40:80],Training_Images_2_list[40:80],Training_Images_7_list[40:80]),axis=0)
    Fold_3_Images=np.concatenate((Training_Images_1_list[80:120],Training_Images_2_list[80:120],Training_Images_7_list[80:120]),axis=0)
    Fold_4_Images=np.concatenate((Training_Images_1_list[120:160],Training_Images_2_list[120:160],Training_Images_7_list[120:160]),axis=0)
    Fold_5_Images=np.concatenate((Training_Images_1_list[160:200],Training_Images_2_list[160:200],Training_Images_7_list[160:200]),axis=0)
    Fold_1_Labels=np.concatenate((Training_Labels_1_list[:40],Training_Labels_2_list[:40],Training_Labels_7_list[:40]),axis=0) 
    Fold_2_Labels=np.concatenate((Training_Labels_1_list[40:80],Training_Labels_2_list[40:80],Training_Labels_7_list[40:80]),axis=0)
    Fold_3_Labels=np.concatenate((Training_Labels_1_list[80:120],Training_Labels_2_list[80:120],Training_Labels_7_list[80:120]),axis=0)
    Fold_4_Labels=np.concatenate((Training_Labels_1_list[120:160],Training_Labels_2_list[120:160],Training_Labels_7_list[120:160]),axis=0)
    Fold_5_Labels=np.concatenate((Training_Labels_1_list[160:200],Training_Labels_2_list[160:200],Training_Labels_7_list[160:200]),axis=0)
    K_Neighbours = [1,3,5,7,9]
    Fold1=[]
    Fold2=[]
    Fold3=[]
    Fold4=[]
    Fold5=[]
    for k in K_Neighbours:
        print("%s Neighbours:" % k)
        Training_Images=np.concatenate((Fold_1_Images,Fold_2_Images,Fold_3_Images,Fold_4_Images),axis=0)
        Training_Labels=np.concatenate((Fold_1_Labels,Fold_2_Labels,Fold_3_Labels,Fold_4_Labels),axis=0) 
        Validation_Images=Fold_5_Images
        Validation_Labels=Fold_5_Labels
        Validation_Prediction = KNN(Training_Images,Training_Labels,Validation_Images,k)
        accuracy1=Accuracy_Rate(Validation_Labels,Validation_Prediction)
        Fold1.append(accuracy1)
        print("Validation Accuracy Fold1: %s" %accuracy1)
        Training_Images=np.concatenate((Fold_2_Images,Fold_3_Images,Fold_4_Images,Fold_5_Images),axis=0)
        Training_Labels=np.concatenate((Fold_2_Labels,Fold_3_Labels,Fold_4_Labels,Fold_5_Labels),axis=0) 
        Validation_Images=Fold_1_Images
        Validation_Labels=Fold_1_Labels
        Validation_Prediction = KNN(Training_Images,Training_Labels,Validation_Images,k)
        accuracy2=Accuracy_Rate(Validation_Labels,Validation_Prediction)
        Fold2.append(accuracy2)
        print("Validation Accuracy Fold2: %s" %accuracy2)
        Training_Images=np.concatenate((Fold_3_Images,Fold_4_Images,Fold_5_Images,Fold_1_Images),axis=0)
        Training_Labels=np.concatenate((Fold_3_Labels,Fold_4_Labels,Fold_5_Labels,Fold_1_Labels),axis=0) 
        Validation_Images=Fold_2_Images
        Validation_Labels=Fold_2_Labels
        Validation_Prediction = KNN(Training_Images,Training_Labels,Validation_Images,k)
        accuracy3=Accuracy_Rate(Validation_Labels,Validation_Prediction)
        Fold3.append(accuracy3)
        print("Validation Accuracy Fold3: %s" %accuracy3)
        Training_Images=np.concatenate((Fold_4_Images,Fold_5_Images,Fold_1_Images,Fold_2_Images),axis=0)
        Training_Labels=np.concatenate((Fold_4_Labels,Fold_5_Labels,Fold_1_Labels,Fold_2_Labels),axis=0) 
        Validation_Images=Fold_3_Images
        Validation_Labels=Fold_3_Labels
        Validation_Prediction = KNN(Training_Images,Training_Labels,Validation_Images,k)
        accuracy4=Accuracy_Rate(Validation_Labels,Validation_Prediction)
        Fold4.append(accuracy4)
        print("Validation Accuracy Fold4: %s" %accuracy4)
        Training_Images=np.concatenate((Fold_5_Images,Fold_1_Images,Fold_2_Images,Fold_3_Images),axis=0)
        Training_Labels=np.concatenate((Fold_5_Labels,Fold_1_Labels,Fold_2_Labels,Fold_3_Labels),axis=0) 
        Validation_Images=Fold_4_Images
        Validation_Labels=Fold_4_Labels
        Validation_Prediction = KNN(Training_Images,Training_Labels,Validation_Images,k)
        accuracy5=Accuracy_Rate(Validation_Labels,Validation_Prediction)
        Fold5.append(accuracy5)
        print("Validation Accuracy Fold5: %s" %accuracy5)
    print "Fold1 Accuarcy :",np.mean(Fold1)
    print "Fold2 Acuuracy:",np.mean(Fold2)
    print "Fold3 Accuracy:",np.mean(Fold3)
    print "Fold4 Accuracy:",np.mean(Fold4)
    print "Fold5 Accuracy:",np.mean(Fold5)
    Test_Images=np.reshape(np.array(Test_images[16:]),(10000,784))
    Test_Labels=np.array(Test_labels[8:])
    Test_Images_1_list=[]
    Test_Labels_1_list=[]
    Test_Images_2_list=[]
    Test_Labels_2_list=[]
    Test_Images_7_list=[]
    Test_Labels_7_list=[]
    for i in range (10000):
        if(Test_Labels[i]==1):
            Test_Labels_1_list.append(Test_Labels[i])
            Test_Images_1_list.append(Test_Images[i]) 
        elif(Test_Labels[i]==2):
            Test_Labels_2_list.append(Test_Labels[i])
            Test_Images_2_list.append(Test_Images[i]) 
        elif(Test_Labels[i]==7):
            Test_Labels_7_list.append(Test_Labels[i])
            Test_Images_7_list.append(Test_Images[i])             
    Test_Images_1_Array=np.array(Test_Images_1_list[:50])
    Test_Labels_1_Array=np.array(Test_Labels_1_list[:50])
    Test_Images_2_Array=np.array(Test_Images_2_list[:50])
    Test_Labels_2_Array=np.array(Test_Labels_2_list[:50])
    Test_Images_7_Array=np.array(Test_Images_7_list[:50])
    Test_Labels_7_Array=np.array(Test_Labels_7_list[:50])
    Test_Images_127=np.concatenate((Test_Images_1_Array,Test_Images_2_Array,Test_Images_7_Array),axis=0)
    Test_Labels_127=np.concatenate((Test_Labels_1_Array,Test_Labels_2_Array,Test_Labels_7_Array),axis=0)
    Training_Images=np.concatenate((Fold_3_Images,Fold_4_Images,Fold_5_Images,Fold_1_Images),axis=0)
    Training_Labels=np.concatenate((Fold_3_Labels,Fold_4_Labels,Fold_5_Labels,Fold_1_Labels),axis=0) 
    Accuracy=[]
    for k in K_Neighbours:
        Test_Prediction = KNN(Training_Images,Training_Labels,Test_Images_127,k)
        accuracy=Accuracy_Rate(Test_Labels_127,Test_Prediction)
        Accuracy.append(accuracy)
        print("%s Neighbours:" % k)
        print("Test Accuracy Fold3: %s" % accuracy)
    print  'Test Accuracy:',np.mean(Accuracy)   
if __name__ == '__main__':
	main()
