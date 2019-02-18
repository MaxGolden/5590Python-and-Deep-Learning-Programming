"""
3 use the SVM with RBF kernel on the same dataset. How the result changed?

"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Use the same dataset above
iris = pd.read_csv('iris.csv')
x = iris[['sepal length', 'sepal width', 'petal length', 'petal width']]
y = iris['class']

# Use cross validation to create training and testing part
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Implement linear SVM method using scikit library
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'.format(svm.score(X_train, y_train)))
# test data set acc
print('Accuracy of SVM classifier on test set: {:.2f}'.format(svm.score(X_test, y_test)))

# How the result changed?

# The performance is getting better than normal SVM, based on my test, the accuracy with RBF kernel is
# 0.02 percent higher than  SVM without RBF kernel.











