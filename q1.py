import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

#function to compute the cost function
def compute_cost(X,Y,theta):
    '''
    Input: X=Design matrix of dim (no of samples, no of features+1 )
           Y=response vector of dim (no of samples,1)
           theta=parameter vector of dim (no of features+1,1)
    Output: cost= mean squared error of the hypothesis function of dim (1,1)
    '''
    #dimension check when X contains a single sample/row(incase of stochastic gradient descent)
    if(X.ndim==1):
        X=X.reshape(1,-1)
        Y=Y.reshape(-1,1)   
    cost=np.sum(np.power(Y-np.dot(X,theta),2))/(2*X.shape[0])   
    cost=cost.reshape(-1,1)
    return cost

#function to update parameters
def update_parameters(X,Y, theta, learning_rate):   
    '''
    Input: X=Design matrix of dim (no of samples, no of features+1)
           Y=response vector of dim (no of samples,1)
           theta=parameter vector of dim (no of features+1,1)
           learning_rate= learning rate is a scalar, dim=0
    Output: theta= updated parameter vector of dim (no of features+1, 1)
    '''
    #dimension check when X contains a single sample/row(incase of stochastic gradient descent)
    if(X.ndim==1):
        X=X.reshape(1,-1)
        Y=Y.reshape(-1,1)    
    error=Y-np.dot(X,theta)
    theta=theta+learning_rate*(1/X.shape[0])*np.transpose(np.sum(error*X,axis=0,keepdims=True))    
    return theta

#function to perform batch gradient descent
def gradient_descent(X,Y,eps,learning_rate): 
    '''
    Input: X=Design matrix of dim (no of samples, no of features+1)
           Y=response vector of dim (no of samples,1)           
           learning_rate= learning rate is a scalar, dim=0
           eps= hreshold value for stopping criterion
    Output: values is a dictionary contains "cost_array"= stores cost, dim(no of iterations,1) and
                        "theta_array"= stores parameter, dim(no of feattures+1,no of iteeraton) and
                        "theta"= stores final parameters, dim(no of features+1,1) and
                        "count"= scalar, stores no of iterations, dim=0
    '''
    #initializing parameters
    theta=np.array([0,0]).reshape(-1,1)
    
    cost_new=compute_cost(X,Y,theta)
    
    #to keep track of no of iterations
    count=0
    
    #arrays to store the cost and parameter values for plotting
    cost_array=np.array([])
    theta_array=np.array([])
    
    
    while True:
        cost_old=cost_new          
        count=count+1
        
        #updating the parameters and computing the cost
        theta=update_parameters(X,Y, theta, learning_rate)
        cost_new=compute_cost(X,Y,theta)
        
        cost_array=np.append(cost_array,cost_new)
        theta_array=np.append(theta_array,theta)        
        
        #check for stopping criterion
        if(np.abs(cost_old-cost_new)<eps):            
            break
    cost_array=cost_array.reshape(-1,1)    
    theta_array=theta_array.reshape(cost_array.shape[0],-1).T
    
    values={"cost_array":cost_array, "theta_array":theta_array,"theta":theta,"count":count}    
    
    return values

#function to print fitted parameter values
def print_values():
    print("Parameter vector is:\n"+str(theta))
    print("No of iterations is: "+str(count))

#function to plot the fitted graph 
def plot_graph(X,Y,theta):
    '''
    Input: X=Design matrix of dim (no of samples, no of features+1)
           Y=response vector of dim (no of samples,1)
           theta=parameter vector of dim (no of features+1,1)
    '''
    #plot the training data points
    plt.scatter(X[:,1],Y,label="sample points",color="red")
    
    #plot the fitted line
    y_hat=theta[0]+theta[1]*X[:,1]    
    plt.plot(X[:,1],y_hat,label= "predicted function")
    
    plt.xlabel('Acidity of wine')
    plt.ylabel('Density of wine')
    plt.title("Fitting data using linear regression and gradient Descent")
    plt.legend(loc="lower right")
    
    plt.show()

#function to plot the contour cost function wrt the parameters
def plot_contour(X,Y,theta0_array,theta1_array):
    '''
    Input: X=Design matrix of dim (no of samples, no of features+1)
           Y=response vector of dim (no of samples,1)
           theta0_array=parameter vector of dim (no of iterations,1),intercept term
           theta1_array=parameter vector of dim (no of features+1,1),corresponding to first feature
    '''
    #creating mesh grid
    x=np.linspace(-0.1,1.2,100)
    y=np.linspace(-0.1,1.3,100)
    x,y=np.meshgrid(x,y)
    
    #reshaping x and y grid for calculation purpose
    grid= np.vstack((x.reshape(1,-1),y.reshape(1,-1)))
    
    #computing cost at the grid points
    cost=np.sum(np.power((Y-np.dot(X,grid)),2),axis=0,keepdims=True)/(2*m)
    cost=cost.reshape(x.shape) 
    
    fig,ax = plt.subplots()       
    plt.xlabel('Acidity of wine')
    plt.ylabel('Density of wine')
    plt.title("Contour plot of the cost function")
    ax.contour(x,y,cost,20)   
    plt.scatter(theta0_array,theta1_array,s=4)
    plt.show()
    
#plot the surface of cost function
def plot_surface(X,Y,theta0_array, theta1_array,cost_array,elev,azim):
    '''
    Input: X=Design matrix of dim (no of samples, no of features+1)
           Y=response vector of dim (no of samples,1)
           theta0_array=parameter vector of dim (no of iterations,1),intercept term
           theta1_array=parameter vector of dim (no of features+1,1),corresponding to first feature
           cost_array=dim(no of iterations,1)
           elev= 3d surface vertical angle view
           azim=3d surface x-y plane angle view
    '''
    x=np.linspace(-1,2.5,100)
    y=np.linspace(-2.5,2,100)
    x,y=np.meshgrid(x,y)
    
    #reshaping x and y grid for calculation purpose
    grid= np.vstack((x.reshape(1,-1),y.reshape(1,-1)))
    
    #computing cost at the grid points
    cost=np.sum(np.power((Y-np.dot(X,grid)),2),axis=0,keepdims=True)/(2*m)
    cost=cost.reshape(x.shape)
    
    fig=plt.figure(figsize=[12,8])
    ax = fig.gca(projection='3d')
    ax.plot_surface(x,y,cost,cmap='viridis', edgecolor='none',alpha=0.6)   
    ax.view_init(elev,azim)
    ax.scatter3D(theta0_array,theta1_array,cost_array,color="black",s=8)
    plt.xlabel('Acidity of wine')
    plt.ylabel('Density of wine')
    ax.set_zlabel('cost function')
    plt.title("Surface plot of the cost function")
    plt.show()

#load data from files
trainX=np.loadtxt("linearX.csv").reshape(-1,1)
X=(trainX-np.mean(trainX))/np.std(trainX)
Y=np.loadtxt("linearY.csv").reshape(-1,1)

m=X.shape[0]
#stacking a column vector of 1's and creating the design matrix, X contains sample as row
X=np.hstack((np.ones((m,1)),X))


parser = argparse.ArgumentParser()
parser.add_argument('--lr',type = float)
parser.add_argument('--eps',type = float)
args = parser.parse_args()

learning_rate = args.lr
eps = args.eps 

values=gradient_descent(X,Y,eps,learning_rate)
    
theta=values["theta"]
count=values["count"]
theta0_array=values["theta_array"][0,:]
theta1_array=values["theta_array"][1,:]
cost_array=values["cost_array"]

print_values()
plot_graph(X,Y,values["theta"])
plot_contour(X,Y,theta0_array,theta1_array)
plot_surface(X,Y, theta0_array.T,theta1_array.T,cost_array,elev=10,azim=100)  
