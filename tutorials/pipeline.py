"""Pipeline for Supervised Learning."""

from sklearn import tree
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split

iris = datasets.load_iris()

x = iris.data
y = iris.target

# Features and labels and partioning into two labels
# X_train, y_train labels for training data
# y_train, y_test labels for testing data
# test_size represents half of the data for testing
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

treeClassifier = tree.DecisionTreeClassifier()
treeClassifier.fit(X_train, y_train)

predictions = treeClassifier.predict(X_test)
print("Predictions using DecisionTree = ", predictions)

# How accurate the classifier was in the training set
# Comparison between true labels and predicting labels
accuracyScore = accuracy_score(y_test, predictions)
print("Accuracy Score using DecisionTree = ", accuracyScore, "\n")

# Use another classifier to get the same behavior as in the Decision Tree Classifier
neighborClassifier = KNeighborsClassifier()
neighborClassifier.fit(X_train, y_train)
predictions = neighborClassifier.predict(X_test)
print("Predictions using KNeighborsClassifier = ", predictions)

accuracyScore = accuracy_score(y_test, predictions)
print("Accuracy Score using KNeighborsClassifier = ", accuracyScore)

# Final notes:
# Interesting Fact: If we want to use different classifiers
# We would need just to change line 32, 33 or 20,21 
# To the new classifier type, since at high-level they use the same interface