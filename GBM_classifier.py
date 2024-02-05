from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, log_loss
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import GradientBoostingClassifier

import sklearn.datasets as datasets
from matplotlib.colors import ListedColormap

data = datasets.make_circles(
    n_samples=100, factor=0.5, noise=0.15, random_state=0)
x, y = data[0], data[1]

# make it imbalance
idx = np.sort(np.append(np.where(y != 0)[0], np.where(y == 0)[0][:-10]))
x, y = x[idx], y[idx]


class CustomGBMclassifier:

    def __init__(self, n_trees, max_depth, learning_rate):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.trees = []

    def fit(self, X_train, y_train):
        F0 = np.log(y_train.mean()/(1 - y_train.mean()))
        self.F0 = np.full(len(y_train), F0)
        Fm = self.F0.copy()

        for _ in range(self.n_trees):
            prev_prob = np.exp(Fm) / (1 + np.exp(Fm))
            residuals = y_train - prev_prob
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X_train, residuals)
            ids = tree.apply(X_train)
            for index in np.unique(ids):
                res_index = ids == index
                num = residuals[res_index].sum()
                den = (prev_prob[res_index]*(1-prev_prob[res_index])).sum()
                gamma = num / den
                Fm[res_index] += self.learning_rate * gamma

                tree.tree_.value[index, 0, 0] = gamma
            self.trees.append(tree)

    def predict_proba(self, X):

        Fm = self.F0

        for i in range(self.n_trees):
            Fm += self.learning_rate * self.trees[i].predict(X)

        return np.exp(Fm) / (1 + np.exp(Fm))


custom_gbm = CustomGBMclassifier(
    n_trees=20,
    learning_rate=0.1,
    max_depth=1
)
custom_gbm.fit(x, y)
custom_gbm_log_loss = log_loss(y, custom_gbm.predict_proba(x))
print(f"Custom GBM Log-Loss:{custom_gbm_log_loss:.15f}")

sklearn_gbm = GradientBoostingClassifier(
    n_estimators=20,
    learning_rate=0.1,
    max_depth=1
)
sklearn_gbm.fit(x, y)
sklearn_gbm_log_loss = log_loss(y, sklearn_gbm.predict_proba(x))
print(f"Scikit-learn GBM Log-Loss:{sklearn_gbm_log_loss:.15f}")
