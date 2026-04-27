from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, predictions):.2f}")

from sklearn.metrics import r2_score, mean_absolute_error

# R2 Score 
print(f"R2 Score: {r2_score(y_test, predictions):.2f}")

# Mean Absolute Error
print(f"Mean Absolute Error: {mean_absolute_error(y_test, predictions):.2f}")