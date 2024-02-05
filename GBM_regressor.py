from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import GradientBoostingRegressor


class customGBMregressor():
    def __init__(self, n_trees, max_depth, learning_rate) -> None:
        self.n_trees = n_trees
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X_train, y_train):
        self.F0 = y_train.mean()

        prediction = self.F0

        for i in range(self.n_trees):
            residuals = y_train - prediction
            reg_tree = DecisionTreeRegressor(max_depth=self.max_depth)
            reg_tree.fit(X_train, residuals)
            prediction += self.learning_rate * reg_tree.predict(X_train)
            self.trees.append(reg_tree)

    def predict(self, X):
        return self.F0 + self.learning_rate * np.sum([tree.predict(X) for tree in self.trees], axis=0)


def create_polynomial_dataframe(n):
    """
    Crea un dataframe con características polinomiales y un target que sigue una relación polinomial con un factor estocástico.

    Args:
        degree (int): Grado del polinomio.

    Returns:
        pandas.DataFrame: El dataframe con características polinomiales y el target.
    """
    np.random.seed(123)

    x = np.linspace(0, n, n+1)
    y = (stats.gamma.pdf(x, a=2, loc=0, scale=17) +
         np.random.normal(0, 0.002, n+1)) * 1000
    x = x.reshape(-1, 1)

    # Generar otros arrays 'x2' y 'x3'
    x2 = np.random.uniform(low=0, high=10, size=len(x))
    x3 = np.random.randn(len(x))

    # Crear dataframe
    df = pd.DataFrame({
        'Feature 1': x.flatten(),
        'Feature 2': x2,
        'Feature 3': x3,
        'Target': y
    })

    return df


data = create_polynomial_dataframe(200)

X_train = data.drop('Target', axis=1)
y_train = data['Target']

sklearn_gbm = GradientBoostingRegressor(
    n_estimators=20, learning_rate=0.1, max_depth=3)
sklearn_gbm.fit(X_train, y_train)

custom_gbm_regressor = customGBMregressor(
    n_trees=20, max_depth=3, learning_rate=0.1)
custom_gbm_regressor.fit(X_train, y_train)

sklearn_gbm_rmse = mean_squared_error(
    data['Target'], sklearn_gbm.predict(X_train), squared=False)
custom_gbm_rmse = mean_squared_error(
    data['Target'], custom_gbm_regressor.predict(X_train), squared=False)

r2_sk = r2_score(data['Target'], sklearn_gbm.predict(X_train))
r2_custom = r2_score(data['Target'], custom_gbm_regressor.predict(X_train))

print(f"RMSE para GBM de sklearn: {sklearn_gbm_rmse}/nR2 para GBM de sklearn: {r2_sk}/nRMSE para GBM personalizado: {custom_gbm_rmse}/nR2 para GBM personalizado: {r2_custom}")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['Feature 1'], data['Feature 2'], data['Target'])
ax.scatter(data['Feature 1'], data['Feature 2'],
           custom_gbm_regressor.predict(X_train))
plt.show()
