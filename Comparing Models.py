import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Elastic Net': ElasticNet(random_state=42),
    'SVR': SVR(kernel='rbf', C=1e3, gamma=0.1),
    'Linear Regression': LinearRegression(),
    'KNN': KNeighborsRegressor(n_neighbors=5),
    'Polynomial Regression': Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression())])
}
accuracy = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy[name] = model.score(X_test, y_test)
plt.figure(figsize=(10, 6))
plt.bar(accuracy.keys(), accuracy.values())
plt.xlabel('Algorithm')
plt.ylabel('Accuracy (%)')
plt.title('Comparison of Machine Learning Algorithms')
plt.show()
