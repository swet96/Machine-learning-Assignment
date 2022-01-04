#importing necessary libraries
import numpy as np     
import matplotlib.pyplot as plt   
import time
from mpl_toolkits import mplot3d
import pandas as pd
import argparse


def compute_cost(X,Y,theta):
    '''
    Input: X= design matrix of dimension (no of samples, no of features+1)
           Y= response vector of dimension (no of samples,1)
           theta= parameter vector of dimension (no of parameters,1)
    Outpt: cost= Mean squared error, a scalar
    '''   
    #dimension check in case of SGD
    if(X.ndim==1):
        X=X.reshape(1,-1)
        Y=Y.reshape(-1,1)    
    
    cost=np.sum(np.power(Y-np.dot(X,theta),2),axis=0,keepdims=True)*(1/(2*X.shape[0]))
    return cost

def update_parameters(X,Y, theta, learning_rate):  
    '''
    Input: X= design matrix of dimension (no of samples, no of features+1)
           Y= response vector of dimension (no of samples,1)
           theta= parameter vector of dimension (no of parameters,1)
           learning_rate=a scalar
    Outpt: theta= updated parameter vector of dimension (no of parameters,1)
    '''   
    if(X.ndim==1):
        X=X.reshape(1,-1)
        Y=Y.reshape(-1,1)    
    error=Y-np.dot(X,theta)
    theta=theta+learning_rate*(1/X.shape[0])*np.transpose(np.sum(error*X,axis=0,keepdims=True))     
    return theta

def plot_graph(values,elev,azim):
    '''
    Input: values= a dictionary conaying theta and theta_array
           elev= 3d surface vertical angle view
           azim=3d surface x-y plane angle view
    '''
    fig = plt.figure(figsize = (18, 10))
    ax = plt.axes(projection ="3d")
    x=values["theta_array"][:,0]
    y=values["theta_array"][:,1]
    z=values["theta_array"][:,2]
    ax.scatter3D(x, y, z, color = "green",s=2)
    ax.view_init(elev,azim)
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    ax.set_zlabel("z-axis")
    plt.title("3D parameter plot")
    plt.show()
    
def stochastic_gradient_descent(X,Y,batch_size,eps):
    
    if batch_size==1:
        k=100
        epochs=30000
    elif batch_size==100:
        k=5
        epochs=20000
    elif batch_size==10000:
        k=2
        epochs=22000
    elif batch_size==1000000:
        k=1
        epochs=25000
        
    print("Batch_size: "+str(batch_size))
    
    learning_rate=0.001
    theta=np.array([0,0,0]).reshape(-1,1)
    old_cost_array=np.array([])
    new_cost_array=np.array([])
    theta_array=np.array([])
    step=0

    old_cost_array=np.append(new_cost_array,compute_cost(X[0,:].reshape(1,-1), Y[0].reshape(-1,1),theta))

    flag=True
    start=time.time()
    while flag:        
        for i in range(int(X.shape[0]/batch_size)):            
            #print(int(X.shape[0]/batch_size))
            X_data=X[i*batch_size:(i+1)*batch_size,:] 
            Y_data=Y[i*batch_size:(i+1)*batch_size]   
      
            step=step+1    
            theta=update_parameters(X_data,Y_data,theta,learning_rate) 
            theta_array=np.append(theta_array,theta)
        
            new_cost=compute_cost(X_data,Y_data,theta)       
            new_cost_array=np.append(new_cost_array,new_cost)  
            if (step%500==0):
                print("Parameters are: \n"+str(theta))     
            
            if (step%k==0):            
                if(np.abs(np.mean(old_cost_array)-np.mean(new_cost_array))<eps or step>=epochs):
                    flag=False
                    end=time.time()
                    time_taken=end-start  
                    print("Error difference: "+str(np.abs(np.mean(old_cost_array)-np.mean(new_cost_array))))
                    print("Error is: "+str(np.mean(new_cost_array)))
                    print("No of. iterations is: "+str(step))
                    print("Time taken: "+str(end-start))
                    print("Parameters are:\n "+str(theta))                                                          
                    break
                old_cost_array=new_cost_array
                new_cost_array=np.array([])
    
    theta_array=theta_array.reshape(-1,theta.shape[0])    
    values={"theta_array":theta_array,"theta":theta}
    return values 

#generates the same random numbers in each iteration
np.random.seed(0)  

#generating normal random variables with specified mean and variance
X1=np.random.normal(3,2,(1000000,1))
X2=np.random.normal(-1,2,(1000000,1))

#X is the design matrix
X=np.hstack((np.ones(X1.shape),X1,X2))

parameters=np.array((3,1,2)).reshape(-1,1)

#Y is the response vector
Y=np.dot(X,parameters)+np.random.normal(0,np.sqrt(2),(1000000,1))


parser = argparse.ArgumentParser()
parser.add_argument('--bs',type = int)
parser.add_argument('--eps',type = float)
args = parser.parse_args()

batch_size = args.bs
eps = args.eps 

values=stochastic_gradient_descent(X,Y,batch_size,eps)

#loading the test data
test_data=pd.read_csv('q2test.csv').to_numpy()
#print(test_data.shape)
testX=test_data[:,0:2]
#print(testX.shape)
testX=np.hstack((np.ones((test_data.shape[0],1)),testX))
testY=test_data[:,2].reshape(-1,1)

error1=2*compute_cost(testX,testY,parameters)
error2=2*compute_cost(testX,testY,values["theta"])
print("Error wrt the learned hypothesis is: "+str(error2))
print("Error wrt the original hypothesis is: "+str(error1))
plot_graph(values,elev=10,azim=100)