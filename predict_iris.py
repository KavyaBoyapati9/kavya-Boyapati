from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Sample prediction (change values to test)
sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = knn.predict(sample)

# Output result
print("Predicted Class:", iris.target_names[prediction][0])
