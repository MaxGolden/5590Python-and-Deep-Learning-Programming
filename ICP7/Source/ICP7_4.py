from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB

twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

tfidf_Vect = TfidfVectorizer()
# tfidf_Vect = TfidfVectorizer(ngram_range=(1, 2))
# tfidf_Vect = TfidfVectorizer(stop_words='english')

X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)

clf = MultinomialNB()
clf.fit(X_train_tfidf, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)

score = round(metrics.accuracy_score(twenty_test.target, predicted), 4)
print("MultinomialNB accuracy is: ", score)

print("============= KNN ============")
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train_tfidf, twenty_train.target)

acc_knn = round(knn.score(X_train_tfidf, twenty_train.target) * 100, 2)
print("KNN accuracy is:", acc_knn / 100)

