#Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Loading data from files
X = np.genfromtxt('q4x.dat') 
Y=pd.read_csv('q4y.dat',header=None).to_numpy()

#normalizing data
X=np.divide(X-np.mean(X,axis=0),np.std(X,axis=0))

#covert the labels to binary values 0(canada) and 1()
Y = np.where(Y =='Alaska', 1,0)


#function to get feature means vector
def feature_mean_vec(X):
    '''
    Input:  X = contains samples as row vector of dim(no of samples,no of feature)
    Output: column vector of mean of features of dim(no of features,1)
    '''
    mean=np.mean(X,axis=0).reshape(-1,1)    
    return mean


#function to get feature covariance matrix
def feature_cov_mat(X,mean):
    '''
    Input: X = feature matrix of dim(no of samples,no of features)
    Output:mean=coulmn vector of feature means of dim(no of features,1)
    '''
    matrix=np.dot((X-mean.T).T,(X-mean.T))/X.shape[0]
    return matrix


#calculates quadratic decision boundry values at sampled points, needed to visualise boundry in two dimension
def quadratic_boundry(prior0,prior1,mean0,mean1,cov_mat0,cov_mat1):
    '''
    input:prior0, prior1= proportion of classes, each a scalar
        mean0,mean1=mean column vectors of classes each of dim(no of features,1)
        cov_mat0,cov_mat1=covariance matrices of features for classes, each of dimension (no of eatures, no of features)
    
    Output: values= returns x and y grid and decision boundry function calculated at these, all have same dimnsion
            
    '''
    #creating mesh grid
    x=np.linspace(-3,3,1000)
    y=np.linspace(-3,3,1000)
    x,y=np.meshgrid(x,y)
    
    #converting mesh grid to appropriate size for further calculation
    grid= np.vstack((x.reshape(1,-1),y.reshape(1,-1))) 
    
    #determinant of the covariance matrices for the two classes
    det0=np.linalg.det(cov_mat0)
    det1=np.linalg.det(cov_mat1)
    
    term0=np.sum(((grid-mean0).T@np.linalg.inv(cov_mat0))*(grid-mean0).T,axis=1).reshape(-1,1)   
    term1=np.sum(np.dot((grid-mean1).T,np.linalg.inv(cov_mat1))*(grid-mean1).T,axis=1).reshape(-1,1)
    const=np.log((prior0*np.sqrt(det1))/(prior1*np.sqrt(det0)))
   
    #values of the decision boundry function at the mesh grid to plot contour 
    boundry=(0.5*(term1-term0)+const).reshape(x.shape[0],x.shape[1])
    
    #concatenating x and y coordinates of grid and the decision boundry calculated at each point
    values=np.array([x,y,boundry])    
    return values


#calculates linear decision boundry values at sampled points, needed to visualise boundry in two dimension
def linear_boundry(prior0,prior1,mean0,mean1,cov_mat):
    '''
    input:prior0, prior1= proportion of classes, each a scalar
    mean0,mean1=mean column vectors of classes each of dim(no of features,1)
    cov_mat=covariance matrix of features, each of dimension (no of eatures, no of features)
    
    Output: values= returns x and y grid and decision boundry function calculated at these, all have same dimnsion
            
    '''
    #creating mesh grid
    x=np.linspace(-3,3,1000)       
    y=np.linspace(-3,3,1000)
    x,y=np.meshgrid(x,y)
    
    #converting mesh grid to appropriate size for further calculation
    grid= np.vstack((x.reshape(1,-1),y.reshape(1,-1)))    
    
    term1=np.dot(np.dot((mean0-mean1).T,np.linalg.inv(cov_mat)),grid)    
    term2=-.5*np.dot((np.dot(mean0.T,np.linalg.inv(cov_mat))),mean0)    
    term3=0.5*np.dot((np.dot(mean1.T,np.linalg.inv(cov_mat))),mean1)    
    const=np.log((prior0)/(prior1))
    
    #values of the decision boundry function at the mesh grid to plot contour 
    boundry=(term1+term2+term3+const).reshape(x.shape[0],x.shape[1])
    
    #concatenating x and y coordinates of grid and the decision boundry calculated at each point
    values=np.array([x,y,boundry])
    
    return values


#function print the parameter values
def print_parameters():    
    print("Mean of class Canada:\n " +str(mean0))
    print("Mean of class Alaska:\n " +str(mean1))
    print("Covariance matrix of class Canada is:\n " +str(cov_mat0))
    print("Covariance matrix of class Alaska is:\n " +str(cov_mat1))
    print("Covariance matrix for in the linear case is:\n " +str(cov_mat))

#function to plot the linear and quadratic boundries
def plot_boundry(prior0,prior1,mean0,mean1,cov_mat0,cov_mat1,cov_mat):
    '''
    input:prior0, prior1= proportion of classes, each a scalar
          mean0,mean1=mean column vectors of classes each of dim(no of features,1)
          cov_mat0,cov_mat1=covariance matrices of features for classes in quadratic case, each of dimension (no of eatures, no of features)
          cov_mat= covariance matrices of features in linear case, each of dimension (no of eatures, no of features)
    '''
    #getting the values of quadratic and linear decision boundry at mesh grid to plot the contour
    values_quad=quadratic_boundry(prior0,prior1,mean0,mean1,cov_mat0,cov_mat1)
    values_linear=linear_boundry(prior0,prior1,mean0,mean1,cov_mat)
    
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot()
    
    #scatter plot of the two classes
    ax.scatter(X[Y[:,0]==1,0],X[Y[:,0]==1,1],c='r',s=6,label="Alaska")
    ax.scatter(X[Y[:,0]==0,0],X[Y[:,0]==0,1],c='b',s=6,label="Canada")
    
    #plotting linear boundry
    ax.contour(values_quad[0],values_quad[1],values_quad[2],[0])
    #plotting quadratic boundry
    ax.contour(values_linear[0],values_linear[1],values_linear[2],[0])
    plt.legend()
    
    plt.xlabel("Growth ring diameter in fresh water")
    plt.ylabel("Growth ring diameter in marine water")
    plt.title("Gaussian Discriminat Analysis")
    plt.show()


#inputs corresponding to the classes 0 and 1
input0=X[Y[:,0]==0]
input1=X[Y[:,0]==1]

#feature mean vector corresponding to the clases 0 and 1
mean0=feature_mean_vec(input0)
mean1=feature_mean_vec(input1)

#no of samples corresponding to classes 0 and 1
size0=input0.shape[0]
size1=input1.shape[0]

#prios/proportions corresponding to classes 0 and 1
prior0=size0/(size0+size1)
prior1=size1/(size0+size1)


#covariance matrices for quadratic case
cov_mat0=feature_cov_mat(input0,mean0)
cov_mat1=feature_cov_mat(input1,mean1)

#covariance matrix for linear case
cov_mat=(cov_mat1*size1+cov_mat0*size0)/(size1+size0)

print_parameters()
plot_boundry(prior0,prior1,mean0,mean1,cov_mat0,cov_mat1,cov_mat)









