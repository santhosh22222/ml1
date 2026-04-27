from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f"SVM Accuracy: {accuracy_score(y_test, predictions):.2f}")