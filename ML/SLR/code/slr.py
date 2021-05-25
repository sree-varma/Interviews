
### usage: python slr.py -filename name -epochs 500 -batchsize 1 -l 1 -lr 0.01 -path ./../results/

### (Original author: Sreedevi Varma)

"""LOADING PYTHON LIBRARIES"""
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import argparse
import pickle as pkl
import matplotlib
import sklearn
import pickle



"""LOADING THE MODEL CLASS"""

from model_class import *

"""LOADING EVALUATION/SPLIT MODULES FROM SCIPY"""
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import train_test_split


"""COMMAND LINE OPTIONS"""

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-filename', '--filename', dest='dat_file', type=str, help='The name of the input file')
parser.add_argument('-epochs', '--number of epochs for training', dest='epochs', type=float, default=500, help='Number of epochs for training')
parser.add_argument('-batchsize', '--size of training data batch', dest='batch_size', type=float, default=1, help='Size of the training data batch')
parser.add_argument('-lr', '--learning rate', dest='lr', type=float, default=.01, help='learning rate for training')
parser.add_argument('-reg', '--lambda', dest='l', type=float, default=.01, help='regularization coefficient')
parser.add_argument('-path','--results path', dest='path', type=str ,default='./../results/',help = 'Path to save the plots')

parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', default=False,help='Print output on progress running script')

args = parser.parse_args()

verbose = args.verbose



dat_file=args.dat_file
epochs=args.epochs
batch_size=args.batch_size
lr=args.lr
l=args.l
path=args.path


"""LOADING MODELS"""

model=StochasticLogisticRegression(dat_file)
print('Basic stats of the input data',model.data.describe())
data=model.data


""""SPLITTING DATAFILE INTO MUTUALLY EXCLUSIVE TRAINING TESTING AND VALIDATION DATA"""

x_train, x_val, y_train, y_val = train_test_split(data.iloc[:,:4],data.iloc[:,-1] , test_size=200)

x_val,x_test, y_val, y_test = train_test_split(x_val,y_val , test_size=100)

x_train=np.array(x_train)
y_train=np.array(y_train)

x_val=np.array(x_val)
y_val=np.array(y_val)

x_test=np.array(x_test)
y_test=np.array(y_test)
print('Dataset split into train, validation and test sets\nPrinting their shapes\n**************\n',x_train.shape,x_val.shape,x_test.shape,y_train.shape,y_val.shape,y_test.shape)



"""FIT THE MODEL """
print('Fitting the model\n*********************')
history=model.fit(x_train,y_train,x_val,y_val,l=1,batch_size=1,epochs=500,lr=0.01)

"""PLOT THE LOSS AND ACCURACY CURVES"""

# We can see how the performance of the model over the different iterations and 
# plot the loss and accuracy values at each epoch.

fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].plot(history['loss'])
ax[0].plot(history['val_loss'])
ax[0].set_ylabel('Loss',fontsize=14)
ax[0].set_xlabel('Epochs',fontsize=14)
ax[0].set_title('Loss curves',fontsize=14)


ax[1].plot(history['acc'])
ax[1].plot(history['val_acc'])
ax[1].set_ylabel('Accuracy',fontsize=14)
ax[1].set_xlabel('Epochs',fontsize=14)
ax[1].set_title('Accuracy curves',fontsize=14)

fig.tight_layout()
print('Saving the loss and accuracy curves\n**********************')
plt.savefig(path+'training_curves.png',dpi=300)
plt.close()




p=model.predict(x_test)



"""PLOT THE NETWORK PERFORMANCE"""


# Similarly, we can plot the overall performance of the network at a classification threshold of 0.5.
# In an ideal situation the histograms should be well separated creating a perfect classifier.


signals = []
backgrounds = []
truth=[]
print(len(y_test))
for i in range(len(y_test)):
    y = y_test[i]
    if y ==1:
        signals.append(p[i])
        truth.append(1)
    if y ==0:
        backgrounds.append(p[i])
        truth.append(0)
        
bins = np.linspace(-.05, 1.05, 15)

print('Saving the network performance plots\n**********************')
plt.hist(signals, bins, alpha=0.4, label='class-1')
plt.hist(backgrounds,bins, alpha=0.4, label='class-0')
plt.title('Network performance',fontsize=14)
plt.legend(loc='upper right')
plt.savefig(path+'network_performance.png',dpi=300)
plt.close()


"""PLOT THE ROC CURVE"""


# Reciever-Operating-Characteristics can also be plotted from the classifier output.
# ROC curve gives the true positive rates and false positive rates for different classificaiton threshold.
# Area under the cure (AUC) is also a metric to calculate the performance.




fprs, tprs, thresholds = roc_curve(y_test, p)

aucs = auc(fprs,tprs)
print ("AUC: ",aucs)

print('Saving the ROC plot\n**********************')
plt.plot([0, 1], [0, 1], '--', color='black')
plt.xlim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot(fprs,tprs, color='darkorange',lw=2, label='ROC curve (area = %0.2f)'  %aucs)
plt.grid(True)
plt.legend(fontsize=10,loc=3)
plt.savefig(path+'roc_curves.png',dpi=300)
plt.close()

"""SAVING THE MODEL"""

# We can save the model as a pickle file

print('Saving the model\n*********************')
import pickle as pkl
with open(path+'saved_model', 'wb') as file:  
    pkl.dump(model, file)


"""LOADING THE MODEL"""
# The model is loaded and evaluated

with open(path+'saved_model', 'rb') as file:  
    saved_model = pkl.load(file)

""" MODEL EVALUATION"""
p=saved_model.predict(x_test)

print ("Accuracy: ",accuracy_score(y_test, np.around(p)))
print("Confusion Matrix: \n", confusion_matrix(y_test,np.around(p)))

""" TEST CASES """

test_model=StochasticLogisticRegression(dat_file)
def testSigmoid():
        assert model.sigmoid(0)==0.5
       # assert model.sigmoid(1)==0.73
def testCost():
        assert model.cost(x_train[:2,:],y_train[:2],np.array(np.zeros(5)),l=1)==0.693
        



