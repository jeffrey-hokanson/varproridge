import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression

class GaussianProcess:
	def __init__(self, **kwargs):
		self.gpr = GaussianProcessRegressor(**kwargs)
		self.lin = LinearRegression()	

	def fit(self, X, y):
		self.lin.fit(X,y)
		y2 = self.lin.predict(X) - y
		self.gpr.fit(X, y2)

	def predict(self, X):
		return self.lin.predict(X) + self.gpr.predict(X)

	def score(self, X, y, norm = False):
		err = np.linalg.norm(self.predict(X) - y)
		if norm:
			err /= np.linalg.norm(y)

		return err
