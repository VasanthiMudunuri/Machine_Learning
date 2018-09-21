import numpy as np
import matplotlib.pyplot as mplot
import scipy
import scipy.stats

class NaiveBayes_Gaussian:
      def __init__(self,Labels,Images):
        self.Labels = np.array(Labels)
        self.Images = np.array(Images)
        self.Mean = np.zeros((Labels, Images), dtype=np.float)
        self.Variance = np.zeros((Labels, Images), dtype=np.float)
      def prediction(self, data): 
        if len(data.shape) == 2:
            return np.array([self.predict(x) for x in data])
        return self.predict(data)  
      def training(self, data, labels):
        NoOfImages = data.shape[0]
        NoOflabels = np.array([(labels == y).sum() for y in range(self.Labels)], dtype=np.float) 
        for y in range(self.Labels):
            sumOfvalues = np.sum(data[n] if labels[n] == y else 0.0 for n in range(NoOfImages))
            self.Mean[y] = sumOfvalues / NoOflabels[y]
        for y in range(self.Labels):
            sumOfvalues = np.sum((data[n] - self.Mean[y])**2 if labels[n] == y else 0.0 for n in range(NoOfImages))
            self.Variance[y] = sumOfvalues / NoOflabels[y]  
      def predict(self, x):
        cost=np.array([5,2,1,0.5,0.2],dtype=np.float)
        predictions=[]
        for i in range(len(cost)):
            results = [self.classProbability(x, y) for y in range(self.Labels)]
            if results[1]>=cost[i]*results[0]:
                predictions.append(0)
            else:
                predictions.append(1)       
        return predictions        
      def classProbability(self, x, y):
          return (-np.sum([self.probabilityDensityFunction(x[d], self.Mean[y][d], self.Variance[y][d]) for d in range(self.Images)]))
      def probabilityDensityFunction(self, x, Mean, Variance):
        epsilon = 1.0e-5
        if Variance < epsilon:
            return 0.0
        return scipy.stats.norm(Mean, Variance).logpdf(x)    
def MNIST():
    training_images_file = open('C:/AI2/train-images.idx3-ubyte', 'rb')
    training_images=training_images_file.read()
    training_images_file.close()
    training_labels_file = open('C:/AI2/train-labels.idx1-ubyte', 'rb')
    training_labels=training_labels_file.read()
    training_images_file.close()
    Training_images=bytearray(training_images)
    Training_labels=bytearray(training_labels)
    Training_Images=np.reshape(np.array(Training_images[16:]),(60000,784))
    Training_Labels=np.array(Training_labels[8:])
    Training_Labels_0_list=[]
    Training_Images_0_list=[]
    Training_Labels_1_list=[]
    Training_Images_1_list=[]
    Training_Labels_2_list=[]
    Training_Images_2_list=[]
    Training_Labels_3_list=[]
    Training_Images_3_list=[]
    Training_Labels_4_list=[]
    Training_Images_4_list=[]
    Training_Labels_6_list=[]
    Training_Images_6_list=[]
    Training_Labels_7_list=[]
    Training_Images_7_list=[]
    Training_Labels_8_list=[]
    Training_Images_8_list=[]
    Training_Labels_9_list=[]
    Training_Images_9_list=[] 
    Training_Images_5_list=[]
    Training_Labels_5_list=[]
    for i in range (60000):
        if(Training_Labels[i]==0):
            Training_Labels_0_list.append(Training_Labels[i])
            Training_Images_0_list.append(Training_Images[i])  
        elif(Training_Labels[i]==1):
            Training_Labels_1_list.append(Training_Labels[i])
            Training_Images_1_list.append(Training_Images[i])
        elif(Training_Labels[i]==2):
            Training_Labels_2_list.append(Training_Labels[i])
            Training_Images_2_list.append(Training_Images[i]) 
        elif(Training_Labels[i]==3):
            Training_Labels_3_list.append(Training_Labels[i])
            Training_Images_3_list.append(Training_Images[i]) 
        elif(Training_Labels[i]==4):
            Training_Labels_4_list.append(Training_Labels[i])
            Training_Images_4_list.append(Training_Images[i]) 
        elif(Training_Labels[i]==6):
            Training_Labels_6_list.append(Training_Labels[i])
            Training_Images_6_list.append(Training_Images[i]) 
        elif(Training_Labels[i]==7):
            Training_Labels_7_list.append(Training_Labels[i])
            Training_Images_7_list.append(Training_Images[i]) 
        elif(Training_Labels[i]==8):
            Training_Labels_8_list.append(Training_Labels[i])
            Training_Images_8_list.append(Training_Images[i]) 
        elif(Training_Labels[i]==9):
            Training_Labels_9_list.append(Training_Labels[i])
            Training_Images_9_list.append(Training_Images[i]) 
        else:
            Training_Labels_5_list.append(Training_Labels[i])
            Training_Images_5_list.append(Training_Images[i])     
    Training_Images_5_Array=np.array(Training_Images_5_list[:900])
    Training_Labels_5_Array=np.array(Training_Labels_5_list[:900])
    Test_Images_5_Array=np.array(Training_Images_5_list[900:1000])
    Test_Labels_5_Array=np.array(Training_Labels_5_list[900:1000])
    Training_Labels_5_Array[Training_Labels_5_Array == 5]=1
    Test_Labels_5_Array[Test_Labels_5_Array == 5]=1 
    Training_Images_0_Array=np.array(Training_Images_0_list[:100])
    Training_Labels_0_Array=np.array(Training_Labels_0_list[:100])
    Training_Images_1_Array=np.array(Training_Images_1_list[:100])
    Training_Labels_1_Array=np.array(Training_Labels_1_list[:100])
    Training_Images_2_Array=np.array(Training_Images_2_list[:100])
    Training_Labels_2_Array=np.array(Training_Labels_2_list[:100])
    Training_Images_3_Array=np.array(Training_Images_3_list[:100])
    Training_Labels_3_Array=np.array(Training_Labels_3_list[:100])
    Training_Images_4_Array=np.array(Training_Images_4_list[:100])
    Training_Labels_4_Array=np.array(Training_Labels_4_list[:100])
    Training_Images_6_Array=np.array(Training_Images_6_list[:100])
    Training_Labels_6_Array=np.array(Training_Labels_6_list[:100])
    Training_Images_7_Array=np.array(Training_Images_7_list[:100])
    Training_Labels_7_Array=np.array(Training_Labels_7_list[:100])
    Training_Images_8_Array=np.array(Training_Images_8_list[:100])
    Training_Labels_8_Array=np.array(Training_Labels_8_list[:100])
    Training_Images_9_Array=np.array(Training_Images_9_list[:100])
    Training_Labels_9_Array=np.array(Training_Labels_9_list[:100])
    Training_Images_Not5_Array=np.concatenate((Training_Images_0_Array,Training_Images_1_Array,Training_Images_2_Array,Training_Images_3_Array,Training_Images_4_Array,Training_Images_6_Array,Training_Images_7_Array,Training_Images_8_Array,Training_Images_9_Array),axis=0)
    Training_Labels_Not5_Array=np.concatenate((Training_Labels_0_Array,Training_Labels_1_Array,Training_Labels_2_Array,Training_Labels_3_Array,Training_Labels_4_Array,Training_Labels_6_Array,Training_Labels_7_Array,Training_Labels_8_Array,Training_Labels_9_Array),axis=0)  
    Training_Labels_Not5_Array[Training_Labels_Not5_Array > 5]=0
    Training_Labels_Not5_Array[Training_Labels_Not5_Array < 5]=0
    Training_Images=np.concatenate((Training_Images_5_Array,Training_Images_Not5_Array),axis=0)
    Training_Labels=np.concatenate((Training_Labels_5_Array,Training_Labels_Not5_Array),axis=0)
    Test_Images_0_Array=np.array(Training_Images_0_list[100:111])
    Test_Labels_0_Array=np.array(Training_Labels_0_list[100:111])
    Test_Images_1_Array=np.array(Training_Images_1_list[100:111])
    Test_Labels_1_Array=np.array(Training_Labels_1_list[100:111])
    Test_Images_2_Array=np.array(Training_Images_2_list[100:111])
    Test_Labels_2_Array=np.array(Training_Labels_2_list[100:111])
    Test_Images_3_Array=np.array(Training_Images_3_list[100:111])
    Test_Labels_3_Array=np.array(Training_Labels_3_list[100:111])
    Test_Images_4_Array=np.array(Training_Images_4_list[100:111])
    Test_Labels_4_Array=np.array(Training_Labels_4_list[100:111])
    Test_Images_6_Array=np.array(Training_Images_6_list[100:111])
    Test_Labels_6_Array=np.array(Training_Labels_6_list[100:111])
    Test_Images_7_Array=np.array(Training_Images_7_list[100:111])
    Test_Labels_7_Array=np.array(Training_Labels_7_list[100:111])
    Test_Images_8_Array=np.array(Training_Images_8_list[100:111])
    Test_Labels_8_Array=np.array(Training_Labels_8_list[100:111])
    Test_Images_9_Array=np.array(Training_Images_9_list[100:111])
    Test_Labels_9_Array=np.array(Training_Labels_9_list[100:111])
    Test_Images_Not5_Array=np.concatenate((Test_Images_0_Array,Test_Images_1_Array,Test_Images_2_Array,Test_Images_3_Array,Test_Images_4_Array,Test_Images_6_Array,Test_Images_7_Array,Test_Images_8_Array,Test_Images_9_Array),axis=0)
    Test_Labels_Not5_Array=np.concatenate((Test_Labels_0_Array,Test_Labels_1_Array,Test_Labels_2_Array,Test_Labels_3_Array,Test_Labels_4_Array,Test_Labels_6_Array,Test_Labels_7_Array,Test_Labels_8_Array,Test_Labels_9_Array),axis=0)  
    Test_Labels_Not5_Array[Test_Labels_Not5_Array > 5]=0
    Test_Labels_Not5_Array[Test_Labels_Not5_Array < 5]=0
    Test_Images=np.concatenate((Test_Images_5_Array,Test_Images_Not5_Array),axis=0)
    Test_Labels=np.concatenate((Test_Labels_5_Array,Test_Labels_Not5_Array),axis=0)
    NoOfLabels = 2 
    NoOfPixels = 28*28
    mnist = NaiveBayes_Gaussian(NoOfLabels, NoOfPixels)
    mnist.training(Training_Images, Training_Labels)
    test_data, labels=Test_Images,Test_Labels
    test_data, labels = test_data, labels
    results = []
    results.append(mnist.prediction(test_data))
    Results=np.transpose(np.array(results))  
    Results=np.reshape(Results,(5,199))
    for j in range(5):          
        print "Accuracy ",j,":", ((Results[j] == labels).mean())*(100.0),'%'
    Counts=[]    
    for k in range(5):
        Counts.append(Calculate_Rates(labels,Results[k]))    
    return np.array(Counts)
def Calculate_Rates(actual,predict):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(predict)): 
        if actual[i]==predict[i]==1:
            TP += 1
    for i in range(len(predict)): 
        if predict[i]==1 and actual[i]!=predict[i]:
            FP += 1
    for i in range(len(predict)): 
        if actual[i]==predict[i]==0:
            TN += 1
    for i in range(len(predict)): 
        if predict[i]==0 and actual[i]!=predict[i]:
            FN += 1
    print TP, FP, TN, FN
    return TP, FP, TN, FN
   
if __name__=="__main__":
    CountofTrueFalse=MNIST() 
    TPR=[]
    FPR=[]
    for i in range(5):
        TPR.append(CountofTrueFalse[i][0]/float(CountofTrueFalse[i][0]+CountofTrueFalse[i][3]))
        FPR.append(CountofTrueFalse[i][1]/float(CountofTrueFalse[i][1]+CountofTrueFalse[i][2]))
    print FPR
    print TPR
    auc=np.trapz(TPR,FPR)
    mplot.title('ROC Curve')
    mplot.plot(FPR, TPR, 'b', label='AUC = %0.2f' % auc)
    mplot.legend(loc = 'lower right')
    mplot.plot([0, 1], [0, 1],'r--')
    mplot.xlim([0, 1])
    mplot.ylim([0, 1])
    mplot.ylabel('True Positive Rate')
    mplot.xlabel('False Positive Rate')
    mplot.show()
