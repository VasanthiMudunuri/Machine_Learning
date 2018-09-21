import numpy as np
import random

def KMeans(K,Training,Mean,ClusterAssignment):
    Distances=np.array([])
    KMean=np.array([])
    HotMatrix=np.array(np.reshape(np.repeat(1,100000),(10000,10)))
    for i in range(0,len(Training)):
        for j in range(0,len(Mean)):
            sumofSquareofDifferences=0
            for k in range(0,784):
                Difference=HotMatrix[i][j]*((Training[i][k]-Mean[j][k])*(Training[i][k]-Mean[j][k]))
                sumofSquareofDifferences=sumofSquareofDifferences+Difference
            Distances=np.append(Distances,sumofSquareofDifferences)  
    FinalDistances=np.reshape(Distances,(len(Training),len(Mean)))
    clusterAssignment=np.argmin(FinalDistances,axis=1) 
    print clusterAssignment
    for m in range(0,len(Training)):
        for n in range(0,len(Mean)):
            if(n!=clusterAssignment[m]):
                HotMatrix[m][n]=0            
    NumberOfMembers=np.sum(HotMatrix,axis=0) 
    for x in range(0,len(Mean)):
        KHotMatrix=HotMatrix[:,x]
        summation=np.zeros(784,)
        for y in range(0,len(Training)):
            ProductofMembers=Training[y]*KHotMatrix[y]
            summation=ProductofMembers+summation            
        kmean=summation/NumberOfMembers[x]
        KMean=np.append(KMean,kmean)
        summation=[]
    KMean=np.reshape(KMean,(K,784))
    if (np.array_equal(ClusterAssignment,clusterAssignment)):
        return HotMatrix,clusterAssignment
    else:
        ClusterAssignment=clusterAssignment
        return KMeans(K,Training,KMean,ClusterAssignment)

def KMeansPlusPlus(K,Training):
    def distance(Centers):
        DistanceSquared = np.array([min([np.linalg.norm(x-c)**2 for c in Centers]) for x in Training])
        return DistanceSquared
    def chooseNextCenter():
        DistanceSquared=distance(Centers)
        Probabilities = DistanceSquared/DistanceSquared.sum()
        CumulativeProbabilities = Probabilities.cumsum()
        RandomNumber = random.random()
        index = np.where(CumulativeProbabilities >= RandomNumber)[0][0]
        return(Training[index])
    Centers = []
    InitialCenter=random.choice(Training)
    Centers.append(InitialCenter)
    while len(Centers) < K:
        Centers.append(chooseNextCenter())  
    return np.resize(Centers,(K,784))
                          
if __name__ == "__main__":
   test_images_file = open('C:/AI2/t10k-images.idx3-ubyte','rb')
   test_images=test_images_file.read()
   test_images_file.close()
   test_labels_file = open('C:/AI2/t10k-labels.idx1-ubyte','rb')
   test_labels=test_labels_file.read()
   test_labels_file.close()
   Test_images=bytearray(test_images)
   Test_labels=bytearray(test_labels)
   Test_Images=np.reshape(np.array(Test_images[16:]),(10000,784))
   Test_Labels=np.array(Test_labels[8:])
   Initial_Mean_Index=np.array(range(500,5500,500))
   Initial_Mean=Test_Images[Initial_Mean_Index]
   ClusterAssignment=np.array([])
   ClusterAssignment=np.append(ClusterAssignment,np.zeros(10000,))
   Hotmatrix,Clusterassignment=KMeans(10,Test_Images,Initial_Mean,ClusterAssignment)
   print Hotmatrix,Clusterassignment
   Kmeansplusplus_Mean_10=KMeansPlusPlus(10,Test_Images)
   print Kmeansplusplus_Mean_10   
   Hotmatrix,Clusterassignment=KMeans(10,Test_Images,Kmeansplusplus_Mean_10,ClusterAssignment)
   print Hotmatrix,Clusterassignment
   Kmeansplusplus_Mean_3=KMeansPlusPlus(3,Test_Images)
   print Kmeansplusplus_Mean_3  
   Hotmatrix,Clusterassignment=KMeans(3,Test_Images,Kmeansplusplus_Mean_3,ClusterAssignment)
   print Hotmatrix,Clusterassignment
   Initial_Mean_Cheating_Index=np.array([3,2,1,18,4,8,11,0,61,7])
   print Test_Labels[Initial_Mean_Cheating_Index]
   Initial_Mean_Cheating=Test_Images[Initial_Mean_Cheating_Index]
   Hotmatrix,Clusterassignment=KMeans(10,Test_Images,Initial_Mean_Cheating,ClusterAssignment)
   print Hotmatrix,Clusterassignment