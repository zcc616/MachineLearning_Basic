import numpy as np

class BayesClassifier():
    def __init__(self,data):
            self.data=data[:,0:len(data[0])-1]
            self.label=data[:,len(data[0])-1]
            self.dataSize=len(data)
            self.attSize=len(data[0])-1
            self.Train()
            self.prediction=[]
            self.errorRate=0

    def prior(self,target):
            count=0
            for x in self.label:
                if x==target:
                    count+=1
            return count/float(self.dataSize)

    def Mean(self,target):
            Sum=np.zeros([1,self.attSize])
            count=0
            for j in range(self.dataSize):
                if self.label[j]==target:
                    count+=1
                    for i in range(self.attSize):
                        Sum[0,i]+=self.data[j,i]

            return Sum/float(count)
    def Sigma2(self,target):
            Sum=np.zeros([1,self.attSize])
            count=0
            for j in range(self.dataSize):
                if self.label[j]==target:
                    count+=1
                    Sum+=(self.data[j,:]-self.mu[target-1])**2
            return Sum/(float(count)-1)

    def Guassian(sefl,M,Var,X):
        return  1/np.sqrt(2*np.pi*Var)*np.exp(-np.power(X-M,2)/(2*Var))

    def Train(self):
            self.mu=np.concatenate((self.Mean(1),self.Mean(2)))
            self.var2=np.concatenate((self.Sigma2(1),self.Sigma2(2)))
            self.prob=np.array((self.prior(1),self.prior(2)))

    def Test(self,testData):
        data=testData[:,0:len(testData[0])-1]
        label=testData[:,len(testData[0])-1]
        prob1,prob2=np.log(self.prob[0]),np.log(self.prob[1])

        for j in range(self.attSize):
                prob1+=np.log(self.Guassian(self.mu[0][j],self.var2[0][j],testData[:,j]))
                prob2+=np.log(self.Guassian(self.mu[1][j],self.var2[1][j],testData[:,j]))

        for i in range(len(testData)):
            if prob1[i]>=prob2[i]:
                self.prediction.append(1)
            else:
                self.prediction.append(2)

        error=0
        for i in range(len(testData)):
            if  label[i]!=self.prediction[i]:
                error+=1
        self.errorRate=error/len(testData)

def main():

    filename = "/Users/chenchenzhang/Documents/MachineLearning_1/glasshw1.csv"
    f = open(filename, 'r')
    data=np.loadtxt(f,delimiter=",")# data (200,11) 1:9 are attributes  10 are label
    result=open("ML1_data.cvs",'w')
    # training using general method
    MyClass=BayesClassifier(data[:,1:11])
    MyClass.Test(data[:,1:11])
    result.write(str(MyClass.prediction)+'\n')
    # trainning using 5-cross validation

    step,errorRate_5cross=len(data)//5,[]
    prediction5=[]
    for i in range(5):
        testData=data[i*step:(i+1)*step,1:11]
        trainData=np.concatenate((data[0:i*step,1:11],data[(i+1)*step:len(data),1:11]))
        MyClass=BayesClassifier(trainData)
        MyClass.Test(testData)
        prediction5+=MyClass.prediction
    result.write(str(prediction5))
# do more on writing file
# github
# explanation
main()
