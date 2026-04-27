from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

iris = load_iris()
# Use columns 1, 2, 3 to predict column 0 (Sepal Length)
X = iris.data[:, 1:] 
y = iris.data[:, 0]  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f"Linear Regression Mean Squared Error: {mean_squared_error(y_test, predictions):.2f}")

from sklearn.metrics import r2_score, mean_absolute_error

# R2 Score 
print(f"R2 Score: {r2_score(y_test, predictions):.2f}")

# Mean Absolute Error
print(f"Mean Absolute Error: {mean_absolute_error(y_test, predictions):.2f}")