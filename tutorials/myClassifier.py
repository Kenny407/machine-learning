import random
from scipy.spatial import distance
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
# from sklearn.neighbors import KNeighborsClassifier

def euc(a, b):
    return distance.euclidean(a, b)

class ScrappyKNN():
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]

# Import data
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split into test and training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

myClassifier = ScrappyKNN()
# Train the classifier with training data
myClassifier.fit(X_train, y_train)

# Test how accurate it is
predictions = myClassifier.predict(X_test)
print("Accuracy Score =", accuracy_score(y_test, predictions))
