import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.preprocessing import RobustScaler
from math import acos, sin, cos, radians
from Traditiuonal_Models import *

# PREPROCESSING DATA

def readDataset(file_name, extension = 'csv'):
	dataset = None

	switcher = {
		'csv': pd.read_csv(file_name + '.csv')
	}

	dataset = switcher.get(extension.lower(), pd.read_csv(file_name + '.csv'))

	return dataset
# end

def splitAttributes(data, input_idx, output_idx = None):
	if output_idx == None:
		output_idx = list(filter(lambda x: x not in input_idx, list(range(data.shape[1]))))
	# end

	X = data.iloc[:, input_idx].values
	y = data.iloc[:, output_idx].values

	return X, y
# end

def splitDataset(data, train, test, validation = 0):
	thresh = np.array([train, train+test]) * data.shape[0]/(train + test + validation)

	train_data = data.iloc[:thresh[0], :].values
	test_data = data.iloc[thresh[0]:thresh[1], :].values
	validation_data = data.iloc[thresh[1]:, :].values

	return train_data, test_data, validation_data
# end

# GEOGRAPHICS FUNCTIONS

# Robson Implementation (approximate Earth radius):
# geo_dist = lambda x, y: 2 * 6372.8 * asin(sqrt(sin(radians((y[0]-x[0])/2)) ** 2 + cos(radians(x[0])) * cos(radians(y[0])) * sin(radians((y[1]-x[1])/2)) ** 2))

# Distance with Equatorial Earth Radius:
geoDist = lambda x, y: acos(sin(radians(x[0]))*sin(radians(y[0]))+cos(radians(x[0]))*cos(radians(y[0]))*cos(radians(x[1]-y[1]))) * 6378.1

euclideanDist = lambda x, y: np.sqrt(np.sum((x-y)**2))

def geodesicDistance(A, b):
	distances = np.array(list(map(lambda x: geoDist(x, b), A)))

	return distances
# end

# EVALL MODELS

def modelPathLoss(model, distances):
	path_loss = np.vectorize(model.pathLoss)

	return path_loss(distances)
# end

# FINGERPRINT FUNCTIONS

def coordPoints(size_km):
	lat_lim = [-8.08, -8.065]
	lon_lim = [-34.91, -34.887]

	left_down = np.array([lat_lim[0], lon_lim[0]])
	left_up = np.array([lat_lim[1], lon_lim[0]])

	right_down = np.array([lat_lim[0], lon_lim[1]])
	right_up = np.array([lat_lim[1], lon_lim[1]])

	# Calculates Variations in Degrees

	y = max(geoDist(left_down, left_up), geoDist(right_down, right_up))
	x = max(geoDist(left_down, right_down), geoDist(left_up, right_up))

	# print(y/size_km, x/size_km)

	d_lat = (size_km * (lat_lim[1] - lat_lim[0])) / y
	d_lon = (size_km * (lon_lim[1] - lon_lim[0])) / x

	lat = np.linspace(lat_lim[0], lat_lim[1], (lat_lim[1]-lat_lim[0])/d_lat)
	lon = np.linspace(lon_lim[0], lon_lim[1], (lon_lim[1]-lon_lim[0])/d_lon)

	return lat, lon
# end


def erbMatrix(model, erb_pos, lat, lon):

	if (len(model) != 2):
		ones = np.ones(len(lat))
		dist_matrix = np.matrix(list(map(lambda x: geodesicDistance(list(zip(lat, x*ones)), erb_pos), lon)))
		matrix = modelPathLoss(model, dist_matrix)
		
		print(".", end = "", flush = True)
	else:
		ones = np.ones(len(lon))
		matrix = np.transpose(np.array(list(map(lambda x: predictModel(model, np.array(list(zip(x*ones, lon)))), lat))))

	return matrix
# end


def pathLossMatrix(model, erb_coord, grid):
	lat, lon = coordPoints(grid)

	lat = (lat[:-1] + lat[1:]) / 2
	lon = (lon[:-1] + lon[1:]) / 2

	print("Calculating matrix", end = "", flush = True)

	if (len(model) != 2):
		matrix = list(map(lambda x: erbMatrix(model, x, lat, lon), erb_coord))
	else:
		matrix = erbMatrix(model, erb_coord, lat, lon)

	print(" finished!")

	return np.transpose(matrix), lat, lon
# end


def localizeCoordinates(matrix, path_loss):
	x, y, z = matrix.shape

	# print(matrix)
	# print(matrix.reshape((x*y, z)))
	
	distances = np.array(list(map(lambda x: euclideanDist(x, path_loss), matrix.reshape((x*y, z)))))

	min_idx = distances.argmin()

	lat_idx = min_idx // y
	lon_idx = min_idx %  y

	# print(matrix.shape)
	# print(lat_idx, lon_idx)
	# print(min_idx, distances.shape)

	return lat_idx, lon_idx
# end


def fingerprint(model, grid, param):

	med_coord, rssi, erb_coord, eirp = param

	# Loss Matrix Calculation

	matrix, lat, lon = pathLossMatrix(model, erb_coord, grid)

	# Estimate Location by Matrix

	error = np.array([])

	print("Calculating error", end = "", flush = True)

	for i in range(len(med_coord)):
		coord = med_coord[i]
		path_loss = eirp - rssi[i]

		x, y = localizeCoordinates(matrix, path_loss)

		error = np.append(error, geoDist(coord, np.array([lat[x], lon[y]])))

		# print("Error", i, ":", error[i])
		if ((i+1)%5 == 0):
			print(".", end = "", flush = True)
	# end

	square_err = np.mean(error**2)

	print(" finished!")

	return square_err

# end


def trainModel(x_train, y_train):
	x_scaler = RobustScaler()
	x_scaled = x_scaler.fit_transform(x_train)

	y_scaler = []
	y_scaled = []

	print("Training classifiers", end = "", flush = True)

	for y in np.transpose(y_train):
		scaler = RobustScaler()

		y_scaler.append(scaler)
		y_scaled.append(scaler.fit_transform(y.reshape((y.size, 1))))
	# end

	kernel = 'rbf'
	C = 1e3
	gamma = 0.1

	classifiers = []

	for y in y_scaled:
		classifiers.append(svm.SVR(kernel = kernel, C = C, gamma = gamma).fit(x_scaled, [x[0] for x in y]))
		print(".", end = "", flush = True)
	# end

	print(" finished!")

	return classifiers, y_scaler + [x_scaler]
# end


def predictModel(model, x):
	classifier, scaler = model

	predicted = []

	for i in range(len(classifier)):
		predict = classifier[i].predict(scaler[-1].transform(x))
		predicted.append(scaler[i].inverse_transform(predict.reshape((-1, 1))))
	# end
		
	print(".", end = "", flush = True)

	predicted = np.concatenate(predicted, axis = 1)

	return predicted
# end


# TEST FUNCTIONS

def testModels(freq = 1800):
	models = []
	models.append(FreeSpace(freq))
	models.append(OkumuraHata(freq))
	models.append(Cost231Hata(freq))
	models.append(Cost231(freq))
	models.append(ECC33(freq))
	models.append(Ericsson(freq))
	models.append(Lee(freq))
	models.append(Sui(freq))

	medicoes = readDataset('Datasets/TrainingData')
	# print("READ TrainingData.csv")
	erbs = readDataset('Datasets/Erbs')
	# print("READ Erbs.csv")

	med_coord, rssi = splitAttributes(medicoes, [0, 1])
	erb_coord, eirp = splitAttributes(erbs, [2, 3], [6])

	distances = np.array(list(map(lambda x: geodesicDistance(erb_coord, x), med_coord)))
	path_loss = np.transpose(eirp) - rssi

	# print(distances)
	# print(path_loss)

	errors = np.array([])

	for model in models:
		model_err = modelPathLoss(model, distances) - path_loss
		mean_sqr = np.mean(model_err ** 2)
		print(type(model).__name__ + ": " + str(mean_sqr))

		errors = np.append(errors, mean_sqr)
	# end

	return models[errors.argmin()]
# end


def main():

	medicoes = readDataset('Datasets/TrainingData')
	erbs = readDataset('Datasets/Erbs')
	testes = readDataset('Datasets/TestLocation')

	med_coord, rssi = splitAttributes(testes, [0, 1])
	erb_coord, eirp = splitAttributes(erbs, [2, 3], [6])

	x_train, y_train = splitAttributes(medicoes, [0, 1])

	classifier, scaler = trainModel(x_train, 55.59 - y_train)

	models = [(classifier, scaler)]
	# models = [FreeSpace(1800), OkumuraHata(1800), Cost231Hata(1800), Cost231(1800), ECC33(1800), Ericsson(1800), Lee(1800), Sui(1800)]
	grids = [5e-3, 10e-3, 20e-3]
	param = (med_coord, rssi, erb_coord, eirp)

	for model in models:
		if (type(model) is tuple):
			print("SVM")
		else:
			print(type(model).__name__)
		for grid in grids:
			err = fingerprint(model, grid, param)
			print(str(grid*1e3) + "m: " + str(err))

main()