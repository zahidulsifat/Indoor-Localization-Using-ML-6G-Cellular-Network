import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm
from math import acos, sin, cos, radians, asin, sqrt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Datasets/medicoes.csv')
df = shuffle(df)
test = pd.read_csv('Datasets/testLoc2.csv')

geo_dist = lambda a, b: acos(sin(radians(a[0])) * sin(radians(b[0])) + cos(radians(a[0])) * cos(radians(b[0])) * cos(
    radians(a[1] - b[1]))) * 6378.1
geo_dist_julia = lambda a, b: 2 * 6372.8 * asin(sqrt(
    sin(radians((b[0] - a[0]) / 2)) ** 2 + cos(radians(a[0])) * cos(radians(b[0])) * sin(
        radians((b[1] - a[1]) / 2)) ** 2))

X = df.iloc[:, 2:].values
y = df.iloc[:, 0:2].values
X_val = test.iloc[:, 2:].values
y_val = test.iloc[:, 0:2].values

# K-Nearest Neighbors (KNN) Regression
k_values = [3, 7, 9, 11]
for k in k_values:
    neigh = KNeighborsRegressor(n_neighbors=k)
    neigh.fit(X, y)
    predictions = neigh.predict(X_val)
    dist_err = np.array(list(map(lambda x: geo_dist(x[0], x[1]), zip(predictions, y_val))))
    err_mean = np.mean(dist_err)
    print(f"KNN (k={k}): Mean Error = {err_mean}")

# Support Vector Regression (SVR)
from sklearn.preprocessing import RobustScaler

rbX = RobustScaler()
X_scaled = rbX.fit_transform(X)

rbY1 = RobustScaler()
y1 = rbY1.fit_transform(y[:, 0:1].reshape(-1, 1))  # Reshape y1 to a 2D array

rbY2 = RobustScaler()
y2 = rbY2.fit_transform(y[:, 1:2].reshape(-1, 1))  # Reshape y2 to a 2D array

C = 1e3  # SVM regularization parameter
svc1 = svm.SVR(kernel='rbf', C=C, gamma=0.1).fit(X_scaled, y1)
svc2 = svm.SVR(kernel='rbf', C=C, gamma=0.1).fit(X_scaled, y2)

svm_pred = np.concatenate((svc1.predict(rbX.transform(X_val)).reshape(-1, 1),
                           svc2.predict(rbX.transform(X_val)).reshape(-1, 1)), axis=1)

predicted = np.concatenate((rbY1.inverse_transform(svm_pred[:, 0].reshape(-1, 1)),
                            rbY2.inverse_transform(svm_pred[:, 1].reshape(-1, 1))), axis=1)

dist_err = np.array(list(map(lambda x: geo_dist(x[0], x[1]), zip(predicted, y_val))))
err_mean = np.mean(dist_err)
print(f"SVR: Mean Error = {err_mean}")


# Random Forest Regression
regr = RandomForestRegressor(random_state=0, n_estimators=1000, oob_score=True)
regr.fit(X, y)
predictions = regr.predict(X_val)
dist_err = np.array(list(map(lambda x: geo_dist(x[0], x[1]), zip(predictions, y_val))))
err_mean = np.mean(dist_err)
print(f"Random Forest: Mean Error = {err_mean}")
