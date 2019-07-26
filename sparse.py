"""Toy sparse approximation response surface"""
# (c) 2017 Jeffrey M. Hokanson (jeffrey@hokanson.us)

import numpy as np
from sklearn.linear_model import LassoCV, LassoLarsCV, lars_path
from scipy.optimize import minimize_scalar, brute
from itertools import product

from numpy.polynomial.polynomial import polyvander, polyder
from numpy.polynomial.legendre import legvander, legder 
from numpy.polynomial.chebyshev import chebvander, chebder
from numpy.polynomial.hermite import hermvander, hermder
from numpy.polynomial.laguerre import lagvander, lagder

from ridge import LegendreTensorBasis

class SparsePolynomial:
	def __init__(self, degree, max_iter = 1000, **kwargs):
		self.degree = degree
		self.kwargs = kwargs
		self.max_iter = max_iter

	def fit(self, X, fX, X_test = None, fX_test = None):
		self.X = np.copy(X)
		self.fX = np.copy(fX)

		#if 'tol' not in self.kwargs:
		#	self.kwargs['tol'] = 1e-8 		

		M, m = X.shape
		self.basis = LegendreTensorBasis(m, self.degree)
		
		# Build basis
		A = self.basis.V(X)
		
		if X_test is None or fX_test is None:
			self.lasso = LassoLarsCV(fit_intercept = False, copy_X = False, **self.kwargs)

			self.lasso.fit(A, fX)
			self.c = self.lasso.coef_
		else:
			# Compute alpha using cross-validation

			# Compute path of coefficients generated by LASSO
			alphas, active, coefs = lars_path(A, fX, copy_X = False, verbose = 2, max_iter = self.max_iter)
			

			# Result of LASSO is piecewise linear; this provides an interpolant for values of intermediate alpha
			# Coefficients need to be in ascending order
			coef = lambda(a): np.array([np.interp([a], alphas[::-1], coefs[j,::-1]) for j in range(coefs.shape[0])])
			
			def objfun(alpha):
				C = coef(alpha).flatten()
				score = float(np.linalg.norm(self.basis.VC(X_test, C) - fX_test)/np.linalg.norm(fX))**2
				print float(alpha), score
				return score

			print "Brute force"	
			bounds = sorted([alphas[-1], alphas[0]])
			alpha_vec = np.linspace(bounds[0], bounds[1], 10)
			if np.abs(alpha_vec[0]) > 0:
				alpha_vec = np.hstack([0, alpha_vec])

			fun_vec = [objfun(alpha) for alpha in alpha_vec]
			k_center = np.argmin(fun_vec)
			k_left = max(0,k_center - 1)
			k_right = min(k_center + 1, len(alpha_vec)-1)
			while k_left > 0:
				if fun_vec[k_left] > fun_vec[k_center]:
					break
				else:
					k_left -= 1
			while k_right < len(alpha_vec)-1:
				if fun_vec[k_right] > fun_vec[k_center]:
					break
				else:
					k_right += 1
			
			bracket = [alpha_vec[k_left], alpha_vec[k_center], alpha_vec[k_right]]
			print "Starting minimization"
			print k_left, k_center, k_right	
			print [alpha_vec[k] for k in [k_left, k_center, k_right]]
			print [fun_vec[k] for k in [k_left, k_center, k_right]]
			try:
				res = minimize_scalar(objfun, bracket, tol =1e-5, options = {'maxiter': 20})
				alpha = res.x
			except:
				print "optimization failed"
				alpha = alpha_vec[np.argmin(fun_vec)]

			self.c = coef(alpha).flatten()

	def predict(self, X):
		return self.basis.VC(X, self.c)

	def score(self, X = None, fX = None, norm = False):
		if X is None and fX is None:
			err = np.linalg.norm(self.predict(self.X) - self.fX)
		else:
			err = np.linalg.norm(self.predict(X) - fX)
		if norm:
			err /= np.linalg.norm(fX)
		return err
