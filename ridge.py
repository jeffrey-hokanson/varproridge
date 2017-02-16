"""Ridge function approximation from function values"""
# (c) 2017 Jeffrey M. Hokanson (jeffrey@hokanson.us)

import numpy as np
from itertools import product
from scipy.linalg import orth, norm
from scipy.linalg import svd
from scipy.misc import comb
from copy import deepcopy

class UnderdeterminedException(Exception):
	pass

def lstsq(A,b):
	return np.linalg.lstsq(A,b)[0]


class MultiIndex:
	"""Specifies a multi-index for a polynomial in the monomial basis of fixed degree 

	"""
	def __init__(self, dimension, degree):
		self.dimension = dimension
		self.degree = degree
		self.iterator = product(range(0, degree+1), repeat = dimension)		


	def __iter__(self):
		return self

	def next(self):
		while True:
			alpha = self.iterator.next()
			if sum(alpha) <= self.degree:
				return alpha

	def __len__(self):
		return int(comb(self.degree + self.dimension, self.degree, exact = True))


def residual(U, X, fX, degree = 1, UX = None):
	V = build_V(U, X, degree, UX)
	c = lstsq(V, fX)	
	r = fX - np.dot(V, c)
	return r	

def build_V(U, X, degree = 1, UX = None):
	N, m = X.shape
	if len(U.shape) == 1:
		U = U.reshape(m, -1)
	m, n = U.shape
	
	if UX is None:
		UX = np.dot(U.T, X.T)
	
	# Number of columns in projector onto the polynomial space

	# Multidimensional Vandermonde matrix
	mi = MultiIndex(n, degree)
	V = np.ones((N, len(mi)))
	for j, alpha in enumerate(mi):
		for k in range(len(alpha)):
			V[:,j] *= UX[k]**alpha[k]

	return V


def build_J(U, X, fX, degree = 1, UX = None,  c = None, r = None, V = None):
	"""

	Parameters
	----------
	c: np.array
		polynomial coefficients V^+fX
	"""
	N, m = X.shape
	if len(U.shape) == 1:
		U = U.reshape(m, -1)
	m, n = U.shape
	
	if UX is None:
		UX = np.dot(U.T, X.T)
	if V is None:
		V = build_V(U, X, degree)
	if c is None:	
		c = lstsq(V, fX)
	if r is None:
		r = fX - np.dot(V, c)


	J1 = np.zeros((N,m,n))
	M = V.shape[1]
	J2 = np.zeros((M,m,n))

	# We precompute the SVD to have access to P_V^perp and V^-
	# via matrix multiplication instead of linear solves 
	Y, s, ZT = svd(V, full_matrices = False) 

	for ell in range(n):
		# Construct the derivative of V with respect to U_{k,ell}
		# Since X[:,k] dependence enters multiplicatively, we construct
		# DVDU minus the X[:,k] factor and then apply it at the end
		DVDU = np.ones(V.shape)
		not_ell = set(list(range(n))) - set([ell])
		for j, alpha in enumerate(MultiIndex(n, degree)):
			
			if alpha[ell] == 0:
				DVDU[:,j] = 0
			else:
				# Product of terms other than q = ell
				for q in not_ell:
					DVDU[:,j] *= UX[q,:]**alpha[q]

				if alpha[ell] > 1:
					DVDU[:,j] *= alpha[ell]*(UX[ell,:]**(alpha[ell] - 1))
		
		for k in range(m):
			DVDU_k = X[:,k,None]*DVDU
			# This is the first term in the VARPRO Jacobian minus the projector out fron
			J1[:, k, ell] = np.dot(DVDU_k, c)
			# This is the second term in the VARPRO Jacobian before applying V^-
			J2[:, k, ell] = np.dot((DVDU_k).T, r) 

	# Project against the range of V
	J1 -= np.tensordot(Y, np.tensordot(Y.T, J1, (1,0)), (1,0))
	# Apply V^- by the pseudo inverse
	J2 = np.tensordot(np.diag(1./s),np.tensordot(ZT, J2, (1,0)), (1,0))
	J = -( J1 + np.tensordot(Y, J2, (1,0)))
	return J


def test_residual():
	""" Test the residual using the true solution
	"""

	def f(x):
		w = np.ones(x.shape)
		w /= np.linalg.norm(w)
		w2 = np.zeros(x.shape)
		w2[0] = 1
		return np.dot(x, w)**3 + np.dot(x, w2)**2 + np.dot(x,w)*np.dot(x, w2) + 10.

	# Generate samples of function
	X = np.random.uniform(size = (100, 5))
	fX = np.array([f(x) for x in X])

	# We setup the right subspace so we should have no residual
	U = np.array([np.ones(5), np.zeros(5)]).T
	U[0,1] = 1
	U = orth(U)
	r = residual(U, X, fX, degree = 3)  
	assert np.all(np.isclose(r, 0))


def test_jacobian():
	""" Test the Jacobian using finite differences
	"""

	def f(x):
		w = np.ones(x.shape)
		w /= np.linalg.norm(w)
		w2 = np.zeros(x.shape)
		w2[0] = 1
		return np.dot(x, w)**3 + np.dot(x, w2)**2 + np.dot(x,w)*np.dot(x, w2) + 10.

	# Generate samples of function
	X = np.random.uniform(size = (100, 5))
	fX = np.array([f(x) for x in X])


	U = orth(np.random.randn(5,2))
	degree = 3
	J = build_J(U, X, fX, degree)

	# Finite difference approximation of the Jacobian
	h = 1e-6	
	J_est = np.zeros(J.shape)
	for k, ell in product(range(U.shape[0]), range(U.shape[1])):
		dU = np.zeros(U.shape)
		dU[k, ell] = h
		J_est[:, k, ell] = (residual(U + dU, X, fX, degree) - residual(U - dU, X, fX, degree))/(2*h)
	
	assert np.all(np.isclose(J, J_est))

def grassmann_gauss_newton(U0, X, fX, degree = 1, disp = False, 
	xtol = 1e-7, ftol = 1e-7, gtol = 1e-10, beta = 1e-4, shrink = 0.5, maxiter = 100, reorth = False,
	step0 = 1., history = False, gauss_newton = True, rtol = 0):
	""" Ridge function approximation


	Parameters
	----------
	U0: np.ndarray 
		Initial subspace estimate
	X: np.ndarray
		Coordiantes for each sample
	fX: np.ndarray
		Function values
	degree: positive integer
		Degree of polynomial on the transformed coordinates
	disp: boolean
		If true, show convergence history
	xtol: float
		Optimization will stop if the change in U is smaller than xtol
	ftol: float
		Optimization will stop if the change in the objective function is smaller than ftol
	gtol: float
		Optimization will stop if the norm of the gradient is smaller than gtol
	maxiter: int
		Maximum number of optimization iterations
	step0: float
		Initial step length
	shrink: float
		How much to shrink the step length during backtracking line search
	gauss_newton: boolean
		If true, use Gauss-Newton, if false, use gradient descent
	reorth: boolean
		Reorthogonalize things against the subspace U
	history: boolean
		If true, return a third ouput: a dictionary where each key is a list residual, subspace U, gradient, etc.  

	"""
	U = orth(U0)
	n, m = U.shape
	N, n2 = X.shape
	assert n == n2, "shapes of the subspace and X must match"
	
	if (degree == 1 and m > 1): # "degree 1 polynomial does not permit a subspace of greater than one dimension"
		raise UnderdeterminedException

	if len(MultiIndex(m, degree)) + n*m >= N:
		raise UnderdeterminedException


	UX = np.dot(U.T, X.T)
	V = build_V(U, X, degree, UX) 	# construct the generalized Vandermonde matrix
	c = lstsq(V, fX)				# polynomial coefficients
	r = fX - np.dot(V, c)			# compute the residual
	norm_r = float(norm(r))
	termination_message = 'maxiter exceeded'
	if history:
		hist = {}
		hist['U'] = []
		hist['residual'] = []
		hist['gradient'] = []
		hist['step-length'] = []

	for it in range(maxiter):
		# build the Jacobian
		J = build_J(U, X, fX, degree, UX = UX, c = c, r = r, V = V)
		G = np.tensordot(J, r, (0,0))	# compute the gradient
		if reorth:
			G -= np.dot(U, np.dot(U.T, G))

		if gauss_newton:
			Y, s, ZT = svd(J.reshape(J.shape[0], -1), full_matrices = False, lapack_driver = 'gesvd')
			# Apply the pseudoinverse
			Delta = np.dot(Y[:,:-m**2].T, r)
			Delta = np.dot(np.diag(1/s[:-m**2]), Delta)
			Delta = -np.dot(ZT[:-m**2,:].T, Delta).reshape(U.shape)
			if reorth:
				Delta -= np.dot(U, np.dot(U.T, Delta))
		else:
			Delta = -G

		alpha = np.dot(G.flatten().T, Delta.flatten())
		grad_norm = np.dot(G.flatten().T, G.flatten())
		
		if grad_norm <= gtol:
			t = 0.
			termination_message = "stopped due to small gradient norm"
			break
		
		if alpha >= 0:
			if disp:
				print "Gauss-Newton step not a descent direction"
			Delta = -G
			alpha = -grad_norm
	

		Y, s, ZT = svd(Delta, full_matrices = False, lapack_driver = 'gesvd')
		UZ = np.dot(U, ZT.T)
		t = step0
		maxiter2 = 20
		for it2 in range(maxiter2):
			# Compute new estimate
			U_new = np.dot(UZ, np.diag(np.cos(s*t))) + np.dot(Y, np.diag(np.sin(s*t)))
			# Enforce orthogonality more strictly than the above expression
			U_new = orth(U_new)
			
			# Compute the new residual
			UX_new = np.dot(U_new.T, X.T)
			V_new = build_V(U_new, X, degree, UX_new)
			c_new = lstsq(V_new, fX)
			r_new = fX - np.dot(V_new, c_new)	
			norm_r_new = float(norm(r_new))
			if norm_r_new <= norm_r + alpha * beta * t: 
				break
			t *= shrink
		

		# Compute distance between U and U_new
		# This will raise an exception if the smallest singular value is greater than one 
		# (hence subspaces numerically equivalent)
		with np.errstate(invalid = 'raise'):
			try:
				dist = np.arccos(svd(np.dot(U_new.T, U), compute_uv = False, overwrite_a = True, lapack_driver = 'gesvd')[-1])
			except FloatingPointError:
				dist = 0.
			

		if it2 == maxiter2-1:
			termination_message = "backtracking line search failed to find a good step"
			break

		# Check convergence criteria
		if (norm_r - norm_r_new)<= ftol:
			if norm_r_new <= norm_r:
				U = U_new
				norm_r = norm_r_new
				c = c_new
			termination_message = "stopped due to small change in residual"
			break

		if norm_r_new <= rtol:
			if norm_r_new <= norm_r:
				U = U_new
				norm_r = norm_r_new
				c = c_new
			termination_message = "stopped due to small residual"
			break

		if dist <= xtol:
			if norm_r_new <= norm_r:
				U = U_new
				norm_r = norm_r_new
				c = c_new
			termination_message = "stopped due to small change in U"
			break

		# copy over values
		U = U_new
		UX = UX_new
		V = V_new
		c = c_new
		r = r_new
		norm_r = norm_r_new
		if history:
			hist['U'].append(U)
			hist['residual'].append(r)
			hist['gradient'].append(G)
			hist['step-length'].append(t)
		if disp:
			print "iter %3d\t |r|: %10.10e\t t: %3.1e\t |g|: %3.1e\t |dU|: %3.1e" %(it, norm_r, t, grad_norm, dist)
	if disp:
		print "iter %3d\t |r|: %10.10e\t t: %3.1e\t |g|: %3.1e\t |dU|: %3.1e" %(it, norm_r_new, t, grad_norm, dist)
		print termination_message

	if history:
		return U, c, hist
	else:
		return U, c 



class PolynomialRidgeApproximation:
	def __init__(self, degree = None, subspace_dimension = None, n_init = 1, **kwargs):
		""" Fit a polynomial ridge function to provided data
		
		Parameters
		----------
		degree: non-negative integer
			Polynomial degree to be fit

		subspace_dimension: non-negative integer
			The dimension on which the polynomial is defined

		n_init: positive integer
			The number of random initializations to preform 
			Large values (say 50) help find the global optimum since
			finding the ridge approximation involves a non-convex optimization problem

		**kwargs:
			Additional arguments are passed to the grassmann_gauss_newton function

		"""
		if subspace_dimension is 0 and degree is None:
			degree = 0
		if subspace_dimension is 1 and degree is None:
			degree = 1
		if degree is 0 and subspace_dimension is None:
			subspace_dimension = 0
		if degree is 1 and subspace_dimension is None:
			subspace_dimension = 1

		if degree is 1 and subspace_dimension > 1:
			raise Exception('Affine linear functions intrinsically only have a 1 dimensional subspace')
		if degree is 0 and subspace_dimension > 0:
			raise Exception('The constant function does not have a subspace associated with it')

		self.degree = degree
		self.subspace_dimension = subspace_dimension
		self.kwargs = kwargs
		self.n_init = n_init

	def fit(self, X, y):
		""" Build ridge function approximation
		"""

		# Special case of fitting a constant
		if self.subspace_dimension == 0 and self.degree == 0:
			self.U = np.zeros((X.shape[1], 0))
			self.c = np.linalg.lstsq(build_V(self.U, X), y)[0]
			return
	
		# Special case of fitting an affine linear fit	
		if self.degree == 1 and self.subspace_dimension == 1:
			# Solve the linear least squares problem
			XX = np.hstack([X, np.ones((X.shape[0],1))])
			b = np.linalg.lstsq(XX, y)[0]
			self.U = orth(b[0:-1].reshape(-1,1))
			self.c = np.array([b[-1], 1])
			return

		if 'U0' in self.kwargs and self.n_init == 1:
			U0 = self.kwargs['U0']
			assert U0.shape[1] == self.subspace_dimension
			kwargs = deepcopy(self.kwargs)
			del kwargs['U0']
		elif self.n_init > 1:
			# If we're going to try multiple subspaces, try the one generated by a linear fit first
			rr = PolynomialRidgeApproximation(degree = 1, subspace_dimension = 1, n_init = 1)
			rr.fit(X, y)
			U0 = rr.U
			if self.subspace_dimension > 1:
				U0 = orth(np.hstack([U0, np.random.randn(U0.shape[0], self.subspace_dimension-1)])) 
			kwargs = self.kwargs
		else:
			U0 = orth(np.random.randn(X.shape[1], self.subspace_dimension))
			kwargs = self.kwargs
	
	
		self.U, self.c = grassmann_gauss_newton(U0, X, y, degree = self.degree, **kwargs)

		# Try other initializations
		if self.n_init > 1:
			res_norm_best = self.score(X, y)
			U_best, c_best = self.U, self.c
		else:
			return

		for it in range(1, self.n_init):
			U0 = orth(np.random.randn(X.shape[1], self.subspace_dimension))
			self.U, self.c = grassmann_gauss_newton(U0, X, y, degree = self.degree, **kwargs)
			res_norm_cur = self.score(X, y)
			if res_norm_cur < res_norm_best:
				U_best, c_best = self.U, self.c
		
		self.U, self.c = U_best, c_best
		
	
	def predict(self, X):
		V = build_V(self.U, X, degree = self.degree) 
		return np.dot(V, self.c)

	def score(self, X, y):
		return np.linalg.norm(self.predict(X) - y, 2)

if __name__ == '__main__':

	# Tests to ensure the residual and Jacobian are calculated correctly
	test_residual()
	test_jacobian()


	# Example of fitting a polynomial to data
	# Here we setup a cubic polynomial on a 2-dimensional space
	def f(x):
		w = np.ones(x.shape)
		w /= np.linalg.norm(w)
		w2 = np.zeros(x.shape)
		w2[0] = 1
		return np.dot(x, w)**3 + np.dot(x, w2)**2 + np.dot(x,w)*np.dot(x, w2) + 10.

	# Generate input/output pairs of this function
	if True:
		X = np.random.uniform(size = (1000, 5))
		fX = np.array([f(x) for x in X])
	else:
		X = np.loadtxt('random1_norm.input')
		fX = np.loadtxt('random1.output')[:,0]

	# Setup the fit function.  
	# Here we've followed scikit-learn's approach where the fit routine 
	# is a class and the meta-parameters like degree and subspace dimensions
	# are arguments passed when initializing the class
	pra = PolynomialRidgeApproximation(degree = 7, subspace_dimension = 1, beta = 1e-4, disp = True, maxiter = 500, ftol = -1, gtol = -1,)

	# The fit function then builds the polynomial ridge approximation using Variable Projection
	# and Grassmann Gauss-Newton
	pra.fit(X, fX)

	# Then to evaluate the fitted polynomial ridge approximation, call the predict function:
	y = pra.predict(X)

	# We then print the error of this approximation.
	# This should be approximately zero if our optimization has succeeded
	# since f(x) has the same model form as its approximation
	print "error", np.linalg.norm(y - fX, 2)/np.linalg.norm(fX,2)
	
	# Alternaively we could have evaluated the least squares mismatch using the score function
	print "score function", pra.score(X, fX)/np.linalg.norm(fX,2)


