import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model,metrics
from sklearn.preprocessing import PolynomialFeatures
"""
 KNN Part starts at line 125
"""
class kNN(): # vectorization
    def __init__(self,k,data,target):
        self.k=k
        self.data=data
        self.target=target
        self.pre=None
    def predict(self,test):
        length=test.shape[0]
        self.pre=np.zeros((length,1))
        for i in range(length):
            d=test[i]-self.data
            d=d*d
            d=np.sum(d,axis=1)
            idx=np.argpartition(d.T,self.k)
            for j in idx[0:self.k]:
                self.pre[i]+=self.target[j][0]
            self.pre[i]=self.pre[i]/self.k

    # print
    def Err(self,target):
        if self.pre is None:
            return

        d=self.pre-target
        d=d*d
        return sum(d)/2

class kNN2(kNN):
    def __init__(self,k,data,target):
        kNN.__init__(self,k,data,target)
    def predict(self,test):
        length=test.shape[0]
        self.pre=np.zeros((length,1))

        for i in range(length):
            d=abs(test[i]-self.data)
            d=np.sum(d,axis=1)+0.01 # to avoid 0
            idx=np.argpartition(d.T,self.k)
            Sum=0
            for j in idx[0:self.k]:
                self.pre[i]+=1/d[j]*self.target[j][0]
                Sum+=1/d[j]
            self.pre[i]=self.pre[i]/Sum
    def Err(self,target):
        return kNN.Err(self,target)




def main():

    def polyRegress(num,x,y,x_test,y_test):
          Max=max(x)
          poly=PolynomialFeatures(degree=num)
          MyEng=linear_model.LinearRegression(fit_intercept=False)
          x=poly.fit_transform(x)
          x_test=poly.fit_transform(x_test)

          MyEng.fit(x,y)

          trainPre=MyEng.predict(x)
          testPre=MyEng.predict(x_test)

          trainErr=len(x)*metrics.mean_squared_error(trainPre,y)/2
          testErr=len(x_test)*metrics.mean_squared_error(testPre,y_test)/2
          t=np.arange(0,Max,1)
          t_poly=poly.fit_transform(t.reshape(-1,1))

          v=np.dot(t_poly,np.transpose(MyEng.coef_))+MyEng.intercept_ #left her
          plt.plot(t,v,'.')

    # import data from file
    filename="/Users/chenchenzhang/Documents/MachineLearning_2/auto_train.csv"
    f =open(filename,'r')
    data=np.loadtxt(f,delimiter=',',skiprows=1)
    f.close()
    filename="/Users/chenchenzhang/Documents/MachineLearning_2/auto_test.csv"
    f =open(filename,'r')
    testData=np.loadtxt(f,delimiter=',',skiprows=1)
    f.close()

    # 1.plot the chart which shows the relation between displacement and mpg
    plt.plot(data[:,0],data[:,2],'ro')

    plt.xlabel('displacement')
    plt.ylabel('mpg')
    plt.show()

    x=data[:,0].reshape(-1,1)
    y=data[:,2].reshape(-1,1)
    x_test=testData[:,0].reshape(-1,1)
    y_test=testData[:,2].reshape(-1,1)


    #2. LinearRegression and its train and test result
    polyRegress(1,x,y,x_test,y_test)


    #3 polynomial Features



    polyRegress(2,x,y,x_test,y_test)
    polyRegress(4,x,y,x_test,y_test)
    polyRegress(6,x,y,x_test,y_test)


    #4 multiple linear LinearRegression
    x=data[:,0:2].reshape(-1,2)
    x_test=testData[:,0:2].reshape(-1,2)
    MyEng=linear_model.LinearRegression()
    MyEng.fit(x,y)

    testPre=MyEng.predict(x_test)
    testErr=len(x_test)*metrics.mean_squared_error(testPre,y_test)/2




    #5 runkNN
    knn1=kNN(1,x,y)
    knn3=kNN(3,x,y)
    knn20=kNN(20,x,y)
    knn1.predict(x_test)
    knn3.predict(x_test)
    knn20.predict(x_test)
    print(knn1.Err(y_test))
    print(knn3.Err(y_test))
    print(knn20.Err(y_test))
    #7 modified runKNN
    knn20=kNN2(20,x,y)
    knn20.predict(x_test)
    print(knn20.Err(y_test))













main()
