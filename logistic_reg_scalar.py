from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# This transforms data to have mean=0 and variance=1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Fit and transform training data
X_test_scaled = scaler.transform(X_test)      # Only transform test data using training mean/std

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Evaluate for Overfitting/Underfitting
train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
test_acc = accuracy_score(y_test, model.predict(X_test_scaled))

print(f"Training Accuracy: {train_acc:.2f}")
print(f"Testing Accuracy: {test_acc:.2f}")

# Overfitting check: If Train >> Test, it's Overfitting.
# Underfitting check: If both are low, it's Underfitting.

# Confusion Matrix & Classification Report
predictions = model.predict(X_test_scaled)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nDetailed Classification Report:")
print(classification_report(y_test, predictions, target_names=iris.target_names))