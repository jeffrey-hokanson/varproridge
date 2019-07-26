import numpy as np
from ridge import LegendreTensorBasis

class GlobalPolynomial:
	def __init__(self, degree = 1):
		self.degree = degree
	
	def fit(self, X, y):
		self.lb = np.min(X, axis = 0)
		self.ub = np.max(X, axis = 0)
		X_scaled = 2*(X - self.lb)/(self.ub - self.lb) - 1
		
		self.basis = LegendreTensorBasis(X_scaled.shape[1], self.degree)
		V = self.basis.V(X_scaled)
		assert V.shape[0] >= V.shape[1], "Problem undetetermined; need at least %d samples" % V.shape[1]
		self.c = np.linalg.lstsq(V, y)[0]

	def predict(self, X):
		X_scaled = 2*(X - self.lb)/(self.ub - self.lb) - 1
		V = self.basis.V(X_scaled)
		return np.dot(V, self.c)
	
	def score(self, X, y, norm = False):
		err = np.linalg.norm(self.predict(X) - y)
		if norm:
			err /= np.linalg.norm(y)

		return err


