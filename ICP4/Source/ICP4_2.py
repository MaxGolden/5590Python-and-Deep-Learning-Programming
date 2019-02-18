"""
2 Implement linear SVM method using scikit library

Use the same dataset above

Which algorithm you got better accuracy? Can you justify why?

"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# Use the same dataset above
iris = pd.read_csv('iris.csv')
x = iris[['sepal length', 'sepal width', 'petal length', 'petal width']]
y = iris['class']

# Use cross validation to create training and testing part
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Implement linear SVM method using scikit library
svm = LinearSVC(random_state=0, tol=1e-5)
svm.fit(X_train, y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'.format(svm.score(X_train, y_train)))
# test data set acc
print('Accuracy of SVM classifier on test set: {:.2f}'.format(svm.score(X_test, y_test)))

# Which algorithm you got better accuracy? Can you justify why?

# Naive Bayes was better than the SVM

# Because Naive Bayes is stronger for snippets than for longer documents.
# And Naive Bayes is better than SVM/logistic regression (LR) with few training cases, Naive Bayes is
# also better with short documents.

# In iris data set, based on my training dataset(60% of all data), which only has 90 cases. So, that is why the
# Naive Bayes had better performance than the SVM due to few training cases.
