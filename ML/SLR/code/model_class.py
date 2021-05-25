import numpy as np
import pandas as pd
import re
import sys

from sklearn.metrics import accuracy_score


class StochasticLogisticRegression(object):
    
       ##
    def __init__(self, dat_file):

        # check valid path to dat file supplied
        try:
            assert re.match(".*\.csv", dat_file)
        except AssertionError:
            print("Invalid (non .csv) file supplied")
            sys.exit(1)

        # get root of file name from path
        if re.match("\w+\.csv", dat_file):
            self.dat_file = "./" + dat_file

        # parse .dat into formatted array
        try:
            self.data = pd.read_csv(dat_file)
        except:
            print("failed to read .csv file {} : invalid formatting?".format(self.dat_file))
            sys.exit(1)
            
            
    def sigmoid(self,x):
        
        try :
            sigmoid_output=1/(1+np.exp(-x))
        except TypeError:
            print( "Invalid input type in sigmoid function. Please provide a numerical input")
            sys.exit(1)
            
        return sigmoid_output

    
    def cost(self,x,y,w,l):
        
        try:
            assert x.shape[1]+1==w.shape[0]
        except AssertionError:
            print("Invalid shape")
            sys.exit(1)
            
            
        z=np.dot(x,w[1:])+w[0] 
        
        p_1=y*np.log(self.sigmoid(z)) 
        p_2=(1-y)*np.log(1-self.sigmoid(z))
        rt=(l/(2*len(x))*sum((w[1:]**2)))
        s=-sum(p_1+p_2)/len(x)
        return (s)+rt         #The loss function -1/m[sum ylog(sigma(z))+(1-y)log(1-sigma(z))]+ lambda/2m sum theta^2
 
    def acc(self,y,y_pred):
        return(accuracy_score(y, np.around(y_pred)))

    def fit(self,x_train, y_train,x_val,y_val,l=1,batch_size=1, epochs=50, lr=0.05):        
        loss = []
        val_loss=[]
        accuracy=[]
        val_acc=[]

        # Random sample is drawn from a gaussian/normal distribution with mean 0 and 0.1 as standard deviation
        # Here the weight matrix contains bias term as well which is given by the first element of the weight matrix.
        
        weights = np.random.normal(0,0.1,x_train.shape[1]+1)  
        print(weights.shape)
        N = len(x_train)


        error=0
        for i in range(epochs):    
            
            """
            For stochastic gradient descent we choose random samples from the training data inorder to evaluate the cost function
            In standard stochastic gradient descent, the number of random sample is chosen as 1.
            In mini batch stochastic gradient descent a batch of small numbers is used to evaluate the cost.
                        
            """
            
            random_idx = np.random.choice(len(x_train), size=min(len(x_train), batch_size), replace = False)

            #validation_idx=np.random.choice(len(x_val), size=min(len(x_val), batch_size), replace = False)
           
            y_hat = self.sigmoid(np.dot(x_train[random_idx], weights[1:])+weights[0]) #The output is caclulated
            r=np.dot(l,weights[1:])#regularization
            r[0]=0  #The zeroth term corresponds to the bias term

            """The weights are updated here using the gradient formula given above."""
            
            
            weights[0]-=lr*(np.dot(1,y_hat - y_train[random_idx]))
            
            weights[1:] -= lr *(np.dot(x_train[random_idx].T,  y_hat - y_train[random_idx]) +r)/len(x_train[random_idx])            
            
            
            
            loss.append(self.cost(x_train, y_train, weights,l)) # We save the loss for the weights in every epoch
            val_loss.append(self.cost(x_val,y_val,weights,l))    
            accuracy.append(self.acc(y_train,self.sigmoid(np.dot(x_train,weights[1:]))+weights[0])) # Accuracy is also saved in every epoch
            val_acc.append(self.acc(y_val,self.sigmoid(np.dot(x_val,weights[1:]))+weights[0]))

            self.weights=weights
            self.loss=loss[i]
            self.val_loss=val_loss[i]
            self.accuracy=accuracy[i]
            self.val_acc=val_acc[i]
            
        return{'acc':accuracy,'val_acc':val_acc,'loss':loss, 'val_loss':val_loss, 'weights':weights}

    def predict(self,X):        
            # Predicting with sigmoid function
            z = np.dot(X, self.weights[1:])+self.weights[0]
            # Returning binary result
            return self.sigmoid(z)
