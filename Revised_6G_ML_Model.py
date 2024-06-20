import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm
from math import acos, sin, cos, radians
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load Datasets
df = pd.read_csv('Datasets/TrainingData.csv')
df = shuffle(df)
test = pd.read_csv('Datasets/TestLocation2.csv.')

# Define distance functions
geo_dist = lambda a, b: acos(sin(radians(a[0]))*sin(radians(b[0])) + cos(radians(a[0]))*cos(radians(b[0]))*cos(radians(a[1]-b[1]))) * 6378.1

# Separate features and target
X = df.iloc[:, 2:].values
y = df.iloc[:, 0:2].values
X_val = test.iloc[:, 2:].values
y_val = test.iloc[:, 0:2].values

# Lists to store mean errors and algorithm names
mean_errors = []
algorithm_names = []

# kNN with different k values
for k in [3, 9, 11]:
    neigh = KNeighborsRegressor(n_neighbors=k)
    neigh.fit(X, y)
    dist_err = np.array(list(map(lambda x: geo_dist(x[0], x[1]), zip(neigh.predict(X_val), y_val))))
    err_mean = np.mean(dist_err)
    mean_errors.append(err_mean)
    algorithm_names.append(f"kNN (k={k})")

# Support Vector Machine (SVM)
### SVM

from sklearn.preprocessing import RobustScaler

rbX = RobustScaler()
X_scaled = rbX.fit_transform(X)
y1 = y[:, 0:1]
y2 = y[:, 1:2]

rbY1 = RobustScaler()
y1 = rbY1.fit_transform(y1)

rbY2 = RobustScaler()
y2 = rbY2.fit_transform(y2)

C = 1e3  # SVM regularization parameter
svc1 = svm.SVR(kernel='rbf', C=C, gamma=0.1).fit(X_scaled, [x[0] for x in y1])
svc2 = svm.SVR(kernel='rbf', C=C, gamma=0.1).fit(X_scaled, [x[0] for x in y2])

svm_pred = svc1.predict(rbX.transform(X_val))
svm_pred = np.reshape(svm_pred, (-1, 1))
y1_pred = rbY1.inverse_transform(svm_pred)

svm_pred = svc2.predict(rbX.transform(X_val))
svm_pred = np.reshape(svm_pred, (-1, 1))
y2_pred = rbY2.inverse_transform(svm_pred)

predicted = np.concatenate((y1_pred, y2_pred), axis=1)

dist_err = np.array(list(map(lambda x: geo_dist(x[0], x[1]), zip(predicted, y_val))))
err_mean = np.mean(dist_err)
mean_errors.append(err_mean)
algorithm_names.append("SVM")

# Random Forest
regr = RandomForestRegressor(random_state=0, n_estimators=1000, oob_score=True)
regr.fit(X, y)
dist_err = np.array(list(map(lambda x: geo_dist(x[0], x[1]), zip(regr.predict(X_val), y_val))))
err_mean = np.mean(dist_err)
mean_errors.append(err_mean)
algorithm_names.append("Random Forest")

# Create a bar graph
plt.bar(algorithm_names, mean_errors)
plt.xlabel('Algorithm')
plt.ylabel('Mean Error')
plt.title('Comparison of Mean Errors by Algorithm')
plt.xticks
plt.tight_layout()

# Show the graph
plt.show()
