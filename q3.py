#Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

#Loading data from files
X=pd.read_csv("logisticX.csv",header=None).to_numpy()
Y=pd.read_csv("logisticY.csv",header=None).to_numpy()

#Normalising input,X contains each sample as a row
X=np.divide(X-np.mean(X,axis=0),np.std(X,axis=0))


#no of samples
m=X.shape[0]

#stack column vector of 1's to create design matrix of dim(no of samples, no of features+1)
X=np.hstack((np.ones((m,1)),X))

#function to calculate sigmoid 
def sigmoid(X,theta):   
    '''
    Input: X= contains samples of dim(no of samples, no of features+1)
           theta= parameter vector, dim(features+1,1)
    Output: value=value of sigmoid, dim(no of samples,1) 
    '''
    value=np.divide(1.0,1.0+np.exp(-np.dot(X,theta)))    
    return value

#function to compute cost function
def compute_cost(X,Y,theta):    
    '''
    Input: X= contains samples of dim(no of samples, no of features+1)
           Y=column vector of dim(no of samples,1), contains class labels as 0 and 1
           theta= parameter vector, dim(features+1,1)
           Output: cost= it is a scalar, dim=0
    '''
    
    cost=-np.divide(1.0,X.shape[0])*np.sum(Y*np.log(sigmoid(X,theta))+(1-Y)*np.log(1-sigmoid(X,theta)))     
    return cost

#function to compute the gradient
def gradient(X,Y,theta):
    '''
    Input: X= contains samples of dim(no of samples, no of features+1)
           Y=column vector of dim(no of samples,1), contains class labels as 0 and 1
           theta= parameter vector, dim(features+1,1)
           Output: grad= contains gradient of cost function wrt parameters, of dim(no of features+1,1)
    '''
    error=sigmoid(X,theta)-Y    
    grad=np.divide(1.0,X.shape[0])*np.dot(X.T,error)    
    return grad

#function to compute the hessian
def hessian(X,Y,theta):
    '''
    Input: X= contains samples of dim(no of samples, no of features+1)
           Y=column vector of dim(no of samples,1), contains class labels as 0 and 1
           theta= parameter vector, dim(features+1,1)
           Output: hess= Hessian of cost function wrt parameters, of dim(no of features+1,no of features+1)
    '''
    a=sigmoid(X,theta)
    b=1-a    
    hess=np.divide(1.0,X.shape[0])* np.dot(X.T,a*b*X)    
    return hess

#function to update parameters
def update_parameters(X,Y,theta, learning_rate):
    '''
    Input: X= contains samples of dim(no of samples, no of features+1)
           Y=column vector of dim(no of samples,1), contains class labels as 0 and 1
           theta= parameter vector, dim(features+1,1)
           learning_rate= a scalar 
           Output: theta= updated value of parameters, of dim(no of features+1,1)
    '''
    grad=gradient(X,Y,theta)
    hess=hessian(X,Y,theta)
    theta=theta-learning_rate*np.dot(np.linalg.inv(hess),grad)
    return theta

#function to implement newton's method
def newton_method(X,Y,eps,learning_rate):  
    '''
    Input: X= contains samples of dim(no of samples, no of features+1)
           Y=column vector of dim(no of samples,1), contains class labels as 0 and 1        
           learning_rate= a scalar, dim=0
           eps= threshold value for stopping criterion,a scalar, dim=0
           Output: values= a dictionary which contains 'theta'=updated value of parameters of
                           dim(no of features+1,1)) and "count"= no of iterations, a scalar          
            
    '''
    #Initialize parameters, of dim(no of features+1,1)
    theta= np.array([0,0,0]).reshape(-1,1)
    
    #variable to keep track of the cost
    cost_new=compute_cost(X,Y,theta)
    
    #to keep track of the no of iterations
    count=0
    
    while True:
        
        cost_old=cost_new          
        count=count+1
        
        #updating the cost and parameters
        theta=update_parameters(X,Y, theta, learning_rate)
        cost_new=compute_cost(X,Y,theta)       
        
        #checking for stopping criterion: difference in subsequent cost function
        if(np.abs(cost_old-cost_new)<eps):           
            break
    
    values={"theta":theta,"count":count}          
    return values


#function to print values
def print_values(values):
    '''
    Input: values= values= a dictionary which contains 'theta'=updated value of parameters of
                           dim(no of features+1,1)) and "count"= no of iterations, a scalar       
    '''
    theta=values["theta"]
    count=values["count"]
    print("Parameter vector is:\n "+str(theta))
    print("No of iterations is: "+str(count))  

def plot_boundry(theta):
    '''
    Input: theta= parameter vector of dim(no of features+1,1)
    '''
    x=np.linspace(-4,3,1000)
    f=(-theta[0]-theta[1]*x)/theta[2]

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot()
    

    #scatter plot of the two classes
    ax.scatter(X[Y[:,0]==1,1],X[Y[:,0]==1,2],c='r',s=5,label="class1")
    ax.scatter(X[Y[:,0]==0,1],X[Y[:,0]==0,2],c='b',s=5,label="Class0")

    plt.legend()
    ax.plot(x,f)
    plt.title("Logistic Regression")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.show()
    
parser = argparse.ArgumentParser()
parser.add_argument('--lr',type = float)
parser.add_argument('--eps',type = float)
args = parser.parse_args()

learning_rate = args.lr
eps = args.eps   
values=newton_method(X,Y,eps,learning_rate)
theta=values["theta"]

print_values(values)
plot_boundry(theta)

