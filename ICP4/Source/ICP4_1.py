"""
1 . Implementing Naïve Bayes method using scikit-learn library

Use iris dataset available in https://umkc.box.com/s/pm3cebmhxpnczi6h87k2lwwiwdvtxyk8

Use cross validation to create training and testing part

Evaluate the model on testing part

"""
# from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.model_selection import train_test_split

# irisdatasets = datasets.load_iris()
# # print(irisdatasets)
# x = irisdatasets.data
# y = irisdatasets.target

# load iris data set
iris = pd.read_csv('iris.csv')
x = iris[['sepal length', 'sepal width', 'petal length', 'petal width']]
y = iris['class']

# Use cross validation to create training and testing part
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

clf = GaussianNB()
clf.fit(X_train, y_train)
# GaussianNB(priors=None, var_smoothing=1e-09)
print('Accuracy of Naive Bayes GaussianNB on training set: {:.2f}'.format(clf.score(X_train, y_train)))
# Evaluate the model on testing part
print('Accuracy of Naive Bayes GaussianNB on test set: {:.2f}'.format(clf.score(X_test, y_test)))
