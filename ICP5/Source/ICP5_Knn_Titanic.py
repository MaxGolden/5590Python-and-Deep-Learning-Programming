import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier


train_df = pd.read_csv('./test_preprocessed_max_train.csv')
test_df = pd.read_csv('./test_preprocessed_max_test.csv')
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test = test_df
# Y_test = test_df["Survived"]

print(train_df[train_df.isnull().any(axis=1)])

# KNN
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, Y_train)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print("KNN accuracy is:", acc_knn / 100)

acc = knn.score(X_train, Y_train)
print("KNN test ----accuracy is:", acc)
