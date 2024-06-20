import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsRegressor
from math import *
from sklearn.model_selection import LeaveOneOut

geo_dist = lambda a, b: acos(sin(radians(a[0]))*sin(radians(b[0]))+cos(radians(a[0]))*cos(radians(b[0]))*cos(radians(a[1]-b[1]))) * 6378.1
geo_dist_julia = lambda a, b: 2 * 6372.8 * asin(sqrt(sin(radians((b[0]-a[0])/2)) ** 2 + cos(radians(a[0])) * cos(radians(b[0])) * sin(radians((b[1]-a[1])/2)) ** 2))

df = pd.read_csv('Datasets/TrainingData.csv')
df = shuffle(df)
test = pd.read_csv('Datasets/TestLocation2.csv')

X = df.iloc[:, 2:].values
y = df.iloc[:, 0:2].values
X_val =	test.iloc[:, 2:].values
y_val = test.iloc[:, 0:2].values

mean = {
	'test_index' : np.array([], dtype=np.int32),
	'error' : np.array([])
}

loo = LeaveOneOut()
for train_index, test_index in loo.split(X):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]

	neigh = KNeighborsRegressor(n_neighbors = 3)
	neigh.fit(X_train, y_train)
	dist_err = np.array(list(map(lambda x: geo_dist_julia(x[0], x[1]), zip(neigh.predict(X_test), y_test))))
	err_mean = np.sqrt(np.mean(dist_err*dist_err))
	
	mean["test_index"] = np.append(mean["test_index"], test_index)
	mean["error"] = np.append(mean["error"], err_mean)

index = mean["test_index"][mean["error"] > 0.100]

X_model_1 = X[index]
y_model_1 = y[index]

index_model_2 = [i for i in range(len(X)) if i not in index]

X_model_2 = X[index_model_2]
y_model_2 = y[index_model_2]

neigh1 = KNeighborsRegressor(n_neighbors = 3)
neigh1.fit(X_model_1, y_model_1)

neigh2 = KNeighborsRegressor(n_neighbors = 3)
neigh2.fit(X_model_2, y_model_2)

pred1 = neigh1.predict(X_val)
pred2 = neigh2.predict(X_val)

pred = (pred1 + pred2)/2
dist_err = np.array(list(map(lambda x: geo_dist_julia(x[0], x[1]), zip(pred, y_val))))

print(np.mean(np.abs(dist_err)))
