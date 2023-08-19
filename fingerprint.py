import numpy as np

class Fingerprint():

	def __init__(self, model, grid, erb_coord):
		setConstants()
		setParameters(model, grid, erb_coord)

		pathLossMatrix()
	# end

# INITIALIZATION FUNCTIONS

	def setConstants(self):
		self.lat_lim = [-8.08, -8.065]
		self.lon_lim = [-34.91, -34.887]

		self.similarity = self.euclideanDistance
		self.distance = self.geodesicDistance
	# end

	def setParameters(self, model, grid, erb_coord):
		self.model = model
		self.grid = grid
		self.erb_coord = erb_coord
	# end

# PATH LOSS MATRIX FUNCTIONS

	def pathLossMatrix(self):
		self.lat, self.lon = axisPoints()
	# end

	def axisPoints(self):
		lat0, lat1 = lat_lim[0], lat_lim[1]
		lon0, lon1 = lon_lim[0], lon_lim[1]

		left_down = np.array([lat0, lon0])
		left_up = np.array([lat1, lon0])

		right_down = np.array([lat0, lon1])
		right_up = np.array([lat1, lon1])

		# Calcula as variações em graus

		y = max(geoDist(left_down, left_up), geoDist(right_down, right_up))
		x = max(geoDist(left_down, right_down), geoDist(left_up, right_up))

		d_lat = (self.grid * (lat1 - lat0)) / y
		d_lon = (self.grid * (lon1 - lon0)) / x

		lat = np.linspace(lat0, lat1, (lat1-lat0)/d_lat)
		lon = np.linspace(lon0, lon1, (lon1-lon0)/d_lon)

		lat = (lat[:-1] + lat[1:]) / 2
		lon = (lon[:-1] + lon[1:]) / 2

		return lat, lon
	# end

	def distanceMatrix(self):
		list_distance = lambda a, b: map(lambda c: self.distance(c, b), a)
		dist_matrix = lambda a: list(map(lambda b: list_distance(zip(self.lat, b*np.ones(len(self.lat))), a), self.lon))

		
	# end

# SIMILARITY FUNCTIONS

	def euclideanDistance(self, x, y):
		return np.sqrt(np.sum((x-y)**2))
	# end

# DISTANCE FUNCTIONS

	def geodesicDistance(self, x, y):
		return 2*6372.8*asin(sqrt(sin(radians((y[0]-x[0])/2))**2 + cos(radians(x[0]))*cos(radians(y[0]))*sin(radians((y[1]-x[1])/2))**2))
	# end

# end