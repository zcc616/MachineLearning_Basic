import numpy as np
import matplotlib.pyplot as plt
from math import e
import math

#1(a)
"""
x=np.linspace(-2,3)
plt.plot(x,16*x**4-32*x**3-8*x**2+10*x+9)
print(np.roots([64,-96,-16,10]))
plt.show()
"""
class GradientNet():# how to pass a function
     def __init__(self,k,x,iteration):
         self.k=k
         self.x=x
         self.iteration=iteration
         self.value=[]
     def cosFun(self,x):
         return 16*x**4-32*x**3-8*x**2+10*x+9
     def DcosFun(self,x):
         return 64*x**3-96*x**2-16*x+10
     def run(self):
         result=[]
         for i in range(self.iteration):
             self.x=self.x-self.k*self.DcosFun(self.x)
             result.append(self.x)
             self.value.append(self.cosFun(self.x))
         return result
class sonarNet():
    def __init__(self,k,iteration,Data,Label,Weight):
        self.k=k
        self.iteration=iteration
        self.Data=Data
        self.Label=Label
        self.Weight=Weight
        self.result=[]
        self.theta=0.5
    def Entropy(self,y,r):
         length=len(y[0])
         for i in range(length):
             if y[0][i]<e**-16:
                 y[0][i]=e**(-16)
             elif 1-y[0][i]<e**(-16):
                  y[0][i]=1-e**(-16)
         return -np.sum(np.multiply(r,np.log(y))+np.multiply(1-r,np.log(1-y)))

    def sigmoid(self,w,x,theta):
            g=np.dot(w,x)+theta
            return 1/(1+np.exp(-g))#1x180

    def run(self):
        y=self.sigmoid(self.Weight,self.Data,self.theta)
        self.result.append(self.Entropy(y,self.Label))
        for i in range(self.iteration):

            m2=np.sum(np.multiply(np.subtract(self.sigmoid(self.Weight,self.Data,self.theta),self.Label),self.Data),axis=1)
            m2_1=np.sum(np.multiply(np.subtract(self.sigmoid(self.Weight,self.Data,self.theta),self.Label),1),axis=1)

            self.Weight=self.Weight-self.k*np.reshape(m2,(1,60))
            self.theta=self.theta-self.k*m2_1
            y=self.sigmoid(self.Weight,self.Data,self.theta)
            self.result.append(self.Entropy(y,self.Label))

    def test(self):
        y=self.sigmoid(self.Weight,self.Data,self.theta)
        error=0
        for i in range(len(y[0])):
            flag=0
            if y[0][i]>0.5:
                flag=1
            if flag!=self.Label[0][i]:
                error+=1
        return error/len(y[0])




class sonarNetReg():
    def __init__(self,k,iteration,Data,Label,Weight,Penalty):
        self.k=k
        self.iteration=iteration
        self.Data=Data
        self.Label=Label
        self.Weight=Weight
        self.result=[]
        self.theta=0.5
        self.Penalty=Penalty
    def Entropy(self,y,r):
         length=len(y[0])
         for i in range(length):
             if y[0][i]<e**-16:
                 y[0][i]=e**(-16)
             elif 1-y[0][i]<e**(-16):
                  y[0][i]=1-e**(-16)
         return -np.sum(np.multiply(r,np.log(y))+np.multiply(1-r,np.log(1-y)))+self.Penalty/2*np.sum(np.power(self.Weight,2))

    def sigmoid(self,w,x,theta):
            g=np.dot(w,x)+theta
            return 1/(1+np.exp(-g))#1x180


    def run(self):
        y=self.sigmoid(self.Weight,self.Data,self.theta)
        self.result.append(self.Entropy(y,self.Label))
        for i in range(self.iteration):

            m2=np.sum(np.multiply(np.subtract(self.sigmoid(self.Weight,self.Data,self.theta),self.Label),self.Data),axis=1)
            m2_1=np.sum(np.multiply(np.subtract(self.sigmoid(self.Weight,self.Data,self.theta),self.Label),1),axis=1)

            self.Weight=self.Weight-self.k*(np.reshape(m2,(1,60))+self.Penalty*self.Weight)
            self.theta=self.theta-self.k*m2_1
            y=self.sigmoid(self.Weight,self.Data,self.theta)
            self.result.append(self.Entropy(y,self.Label))
        #elf.pre=self.sigmoid(self.Weight,self.Data,self.theta)
    def test(self):
        y=self.sigmoid(self.Weight,self.Data,self.theta)
        error=0
        for i in range(len(y[0])):
            flag=0
            if y[0][i]>0.5:
                flag=1
            if flag!=self.Label[0][i]:
                error+=1
        return error/len(y[0])

    def testCross(self,testData,testLabel):
         y=self.sigmoid(self.Weight,testData,self.theta)
         error=0
         for i in range(len(y[0])):
             flag=0
             if y[0][i]>0.5:
                 flag=1
             if flag!=testLabel[0][i]:
                 error+=1
         return error/len(y[0])








def main():

     m=GradientNet(0.001,-1,1000)
     res=m.run()
     print("\n\nlearning rate=0.001,X starts at -1:")
     print("First 5 x:",res[0:5])
     print("f(x):",m.value[0:5])
     print("Last 5 x:",res[-5:])
     print("f(x):",m.value[-5:])




     m=GradientNet(0.001,2,1000)
     res=m.run()
     print("\n\nlearning rate=0.001,X starts at 2:")
     print("First 5 x:",res[0:5])
     print("f(x):",m.value[0:5])
     print("Last 5 x:",res[-5:])
     print("f(x):",m.value[-5:])



     m=GradientNet(0.01,-1,1000)
     res=m.run()
     print("\n\nlearning rate=0.01,X starts at -1:")
     print("First 5 x:",res[0:5])
     print("f(x):",m.value[0:5])
     print("Last 5 x:",res[-5:])
     print("f(x):",m.value[-5:])


     m=GradientNet(0.05,-1,100)
     try:
         res=m.run()
         print("\nlearning rate=0.05,X starts at -1:")
         print("First 5 x:",res[0:5])
         print("f(x):",m.value[0:5])
         print("Last 5 x:",res[-5:])
         print("f(x):",m.value[-5:])
     except:
         print("x is too large")

    ############################################### where is relevant
     filename="/Users/chenchenzhang/Documents/MachineLearning_3/sonar.csv"
     f =open(filename,'r')
     data=np.loadtxt(f,delimiter=',',usecols=range(0,60))
     data=data.T #60*180
     f.close()
     f =open(filename,'r')
     label0=np.loadtxt(f,dtype=np.str,delimiter=',',usecols=[60]) #1*180
     f.close()
     label0=np.reshape(label0,(1,-1))
     label=np.ones([1,180])
     for i in range(len(label0[0])):
         if label0[0][i]=="Rock":
             label[0][i]=0

     LRate=[0.001,0.01,0.05,0.1,0.5,1,1.5]
     Weight=np.reshape(np.array([0.5]*60),(1,-1)) # 1*60
     Entropy=[]
     TrainErr=[]
     weight=[]
     for i in range(len(LRate)):
         Net=sonarNet(LRate[i],50,data,label,Weight)
         Net.run()
         Entropy.append(Net.result[-1])
         TrainErr.append(Net.test())
         weight.append(np.power(np.sum(np.power(Net.Weight,2)),0.5))

     print("Entropy: ",Entropy)
     print("TrainErr: ",TrainErr)
     print("L2 norm: :  ",weight)
     print("\n\n")
################################################
     f =open(filename,'r')
     data=np.loadtxt(f,delimiter=',',usecols=range(0,60))
     data=data.T #60*180
     f.close()
     f =open(filename,'r')
     label0=np.loadtxt(f,dtype=np.str,delimiter=',',usecols=[60]) #1*180
     label0=np.reshape(label0,(1,-1))
     label=np.ones([1,180])
     for i in range(len(label0[0])):
         if label0[0][i]=="Rock":
             label[0][i]=0

     Penalty=[0,0.05,0.1,0.2,0.3,0.4,0.5]
     Weight=np.reshape(np.array([0.5]*60),(1,-1)) # 1*60
     Entropy=[]
     TrainErr=[]
     TrainErrCro=[]
     weight=[]
     for i in range(len(Penalty)):
         Net=sonarNetReg(0.001,50,data,label,Weight,Penalty[i])
         Net.run()
         Entropy.append(Net.result[-1])
         TrainErr.append(Net.test())
         weight.append(np.power(np.sum(np.power(Net.Weight,2)),0.5))
        # plt.subplot(2,4,i+1)
        # plt.plot(Net.result)
     print("L2 norm: :",weight)

     for i in range(len(Penalty)):

         step,errorRate=len(data[0])//5,[]
         for j in range(5):
             testData=data[:,j*step:(j+1)*step]
             testLabel=label[:,j*step:(j+1)*step]
             trainData=np.concatenate((data[0:60,0:j*step],data[0:60,(j+1)*step:len(data[0])]),axis=1)
             trainLabel=np.concatenate((label[:,0:j*step],label[:,(j+1)*step:len(label[0])]),axis=1)
             Net=sonarNetReg(0.001,50,trainData,trainLabel,Weight,Penalty[i])
             Net.run()
             errorRate.append(Net.testCross(testData,testLabel))

         TrainErrCro.append(sum(errorRate)/5)


     print("Entropy: ",Entropy)
     print("TrainErr: ",TrainErr)
     print("TrainErrCro: :",TrainErrCro)
main()
