# This a reworking of Paul's code to expose a similar API to mine for comparison purposes

import numpy as np
import pandas as pn
from active_subspaces.utils.response_surfaces import PolynomialApproximation

from pymanopt.manifolds import Grassmann
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, TrustRegions


def _res(U, X, f, rs):
	M, m = X.shape
	Y = np.dot(X, U)
	g = rs.predict(Y)[0]
	res = 0.5*np.linalg.norm(f - g)**2
	return res
	
def _dres(U, X, f, rs):
	M, m = X.shape
	n = U.shape[1]
	Y = np.dot(X, U)
	g, dg = rs.predict(Y, compgrad=True)
	dR = np.zeros((m, n))
	for i in range(M):
		dR += (g[i,0] - f[i,0])*np.dot(X[i,:].reshape((m, 1)), dg[i,:].reshape((1, n)))
	return dR

def RidgeAlternating(X, f, U0, degree=1, maxiter=100, tol=1e-10, history = False, disp = False, gtol = 1e-6, inner_iter = 20):
	if len(f.shape) == 1:
		f = f.reshape(-1,1)

	# Instantiate the polynomial approximation
	rs = PolynomialApproximation(N=degree)
	
	# Instantiate the Grassmann manifold		
	m, n = U0.shape
	manifold = Grassmann(m, n)
	
	if history:
		hist = {}
		hist['U'] = []
		hist['residual'] = []
		hist['inner_steps'] = []
	
	# Alternating minimization
	i = 0
	res = 1e9
	while i < maxiter and res > tol:
		
		# Train the polynomial approximation with projected points
		Y = np.dot(X, U0)
		rs.train(Y, f)
		
		# Minimize residual with polynomial over Grassmann
		func = lambda y: _res(y, X, f, rs)
		grad = lambda y: _dres(y, X, f, rs)
		
		problem = Problem(manifold=manifold, cost=func, egrad=grad, verbosity=0)
		if history:
			solver = SteepestDescent(logverbosity=1, mingradnorm = gtol, maxiter = inner_iter, minstepsize = tol)
			U1,log = solver.solve(problem, x=U0)
		else:
			solver = SteepestDescent(logverbosity=0, mingradnorm = gtol, maxiter = inner_iter, minstepsize = tol)
			U1 = solver.solve(problem, x=U0)
	
		# Evaluate and store the residual
		res = func(U1) 		# This is the squared mismatch
		if history:
			hist['U'].append(U1)
			# To match the rest of code, we define the residual as the mismatch
			r = (f - rs.predict(Y)[0]).flatten()
			hist['residual'].append(r)
			hist['inner_steps'].append(log['final_values']['iterations'])
		if disp:
			print "iter %3d\t |r| : %10.10e" % (i, np.linalg.norm(res))	
		# Update iterators
		U0 = U1
		i += 1
		
	# Store data
	if i==maxiter:
		exitflag = 1
	else:
		exitflag = 0
	
	if history:
		return U0, hist
	else:
		return U0

