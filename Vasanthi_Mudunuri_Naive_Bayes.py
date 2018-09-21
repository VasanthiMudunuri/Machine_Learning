import numpy as np

class NaiveBayes:
    def MNIST(self,training=True):
        if training:
            Images_File = open('C:/AI2/train-images.idx3-ubyte', 'rb')
            Labels_File = open('C:/AI2/train-labels.idx1-ubyte', 'rb')
            images=Images_File.read()
            labels=Labels_File.read()
            Training_images=bytearray(images)
            Training_labels=bytearray(labels)
            Images=np.reshape(np.array(Training_images[16:]),(60000,784))
            Labels=np.array(Training_labels[8:])
        else:
            Images_File = open('C:/AI2/t10k-images.idx3-ubyte','rb')
            Labels_File = open('C:/AI2/t10k-labels.idx1-ubyte','rb')
            images=Images_File.read()
            labels=Labels_File.read()
            Test_images=bytearray(images)
            Test_labels=bytearray(labels)
            Images=np.reshape(np.array(Test_images[16:]),(10000,784))
            Labels=np.array(Test_labels[8:])
        self.Labels = Labels
        self.Images = Images
        return Images, Labels
    def training(self, Images, Labels, Load):
            self.probabilityOfClass = [0 for i in range(10)]
            self.classes = [i for i in range(10)]
            self.probabilityOfPixelGivenClass = [[0 for i in range(28*28)] for j in range(10)]
            
            for i in range(len(Labels)):
                self.probabilityOfClass[Labels[i]] += 1 
                for j in range(len(Images[i])):
                    if Images[i][j] > 25:
                        self.probabilityOfPixelGivenClass[Labels[i]][j] += 1 
            for i in range(len(self.probabilityOfPixelGivenClass)):
                for j in range(len(self.probabilityOfPixelGivenClass[i])):  
                    self.probabilityOfPixelGivenClass[i][j] = (self.probabilityOfPixelGivenClass[i][j]+1)/(float(self.probabilityOfClass[i])+10)   
                self.probabilityOfClass[i] /= float(len(Images))  
            for i in range(len(self.probabilityOfClass)):
                print 'Probability(Class=' + str(i) + ') = ' + str(self.probabilityOfClass[i])[:5]  
    def predict(self,Image):
        max = [0.0 for i in range(10)]
        for i in range(len(self.classes)):
            predict = 0.0
            for j in range(len(Image)):
                if Image[j] > 25:
                    predict += np.log(self.probabilityOfPixelGivenClass[i][j])
                else:
                    predict += np.log(1-self.probabilityOfPixelGivenClass[i][j])
            predict *= np.log(self.probabilityOfClass[i])  
            max[i] = predict
        return np.argmin(max)
if __name__ == '__main__':
    training = True
    nb = NaiveBayes()
    training_images, training_labels = nb.MNIST(training)
    test_images, test_labels = nb.MNIST(not training)
    nb.training(training_images, training_labels, False)
    result = []
    for i in range(len(test_images)):
        result.append(nb.predict(test_images[i]))
    accuracy=0
    for i in range(len(result)):
        if result[i] == test_labels[i]:
            accuracy += 1
    accuracy /= float(len(result)) 
    print ' '
    print 'Accuracy = ' + str(accuracy * 100) + '%\n' 
        