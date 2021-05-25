# Stochastic Logistic Regression

## Introduction


The code is to generate a logistic regression classifier with stochastic gradient descent algorithm.


## Folder Structure
    
    This zip folder contains 3 directories viz. data, code and results.
    
    1. data: Contains the data file used for training, validation and testing.
    2. results: Directory to save the results from the code. The model is also saved in this directory by default.
    3. code: Directory containing model_class.py and slr.py. model_class.py is the python class file containing the StochasticLogisticRegression class. The 		  model is used in the slr.py file where the classifier is trained.
   
## Software Dependencies

The libraries and the versions included in the classifier are,

    python==3.8.3
    numpy==1.19.0
    padas==1.0.5
    matplotlib==3.2.2
    scikit-learn==0.23.1
    pickle==4.0

## Usage:

The code takes command line arguments,

    filename - Name of the input file. 
    epochs - Number of iterations for entire dataset. Default value is 500
    batchsize - Number of training samples in the batch. Default value is 1 (standard stochastic gradient descent), increasing this number will result in a  			minibatch gradient descent method.
    lr- Learning rate for the training. Default value is 0.01
    l- Regularization parameter. Default value is 1
    path- Path for the results to be saved. Default path is ./../results
    
 Usage: python slr.py -filename name -epochs 100 -batchsize 1 -l 1 -lr 0.01 -path ./../results
 
## Results:

Using the model built various plots are produced.

	training curves - Loss and Accuracy of the model is plotted for every epoch.
	network performance - Histograms are plotted for a correctly predicted outputs with a classification threshold of 0.5.
 	roc curve - ROC curve is plotted for the model. This plots the true positive rates and false positive rates for various classificaiton threshold.
The accuracy of the current model is 0.82.
The Area under the curve (AUC) of the ROC curve is 0.924
The confusion matrix is obtained as,
[[37  10]
 [8  45]]
## Notebook and References

A jupyter notebook is enclosed within the zip file along with the references used to build the codebase.





```python

```
