"""Ridge function approximation from function values"""
# (c) 2017 Jeffrey M. Hokanson (jeffrey@hokanson.us)

import numpy as np
from itertools import product
from scipy.linalg import orth, norm
from scipy.linalg import svd
from scipy.misc import comb
from copy import deepcopy
from numpy.polynomial.polynomial import polyvander, polyder
from numpy.polynomial.legendre import legvander, legder 
from numpy.polynomial.chebyshev import chebvander, chebder
from numpy.polynomial.hermite import hermvander, hermder
from numpy.polynomial.laguerre import lagvander, lagder


# Symbolic integration 
from sympy import Symbol, integrate, sqrt, diff, lambdify, Poly
from sympy.matrices import Matrix, zeros


# Caching for Orthogonal basis

class UnderdeterminedException(Exception):
	pass

class IllposedException(Exception):
	pass

def lstsq(A,b):
	return np.linalg.lstsq(A,b)[0]


def _full_index_set(n, d):
	"""
	A helper function for index_set.
	"""
	if d == 1:
		I = np.array([[n]])
	else:
		II = _full_index_set(n, d-1)
		m = II.shape[0]
		I = np.hstack((np.zeros((m, 1)), II))
		for i in range(1, n+1):
			II = _full_index_set(n-i, d-1)
			m = II.shape[0]
			T = np.hstack((i*np.ones((m, 1)), II))
			I = np.vstack((I, T))
	return I

def index_set(n, d):
	"""Enumerate multi-indices for a total degree of order `n` in `d` variables.
	Parameters
	----------
	n : int
		degree of polynomial
	d : int
		number of variables, dimension
	Returns
	-------
	I : ndarray
		multi-indices ordered as columns
	"""
	I = np.zeros((1, d))
	for i in range(1, n+1):
		II = _full_index_set(i, d)
		I = np.vstack((I, II))
	return I[:,::-1]




class MultiIndex:
	"""Specifies a multi-index for a polynomial in the monomial basis of fixed degree 

	"""
	def __init__(self, dimension, degree):
		self.dimension = dimension
		self.degree = degree
		#self.iterator = product(range(0, degree+1), repeat = dimension)	
		idx = index_set(degree, dimension).astype(int)
		self.iterator = iter(idx)

	def __iter__(self):
		return self

	def next(self):
		return self.iterator.next()
		#while True:
		#	alpha = self.iterator.next()
		#	if sum(alpha) <= self.degree:
		#		return alpha

	def __len__(self):
		return int(comb(self.degree + self.dimension, self.degree, exact = True))

class Basis:
	pass

class TensorBasis(Basis):
	def __init__(self, n, p):
		self.n = n
		self.p = p
		self.vander = polyvander
		self.der = polyder
		self.indices = index_set(p, dimension).astype(int)

		self.build_Dmat()
		

	def build_Dmat(self):
		self.Dmat = np.zeros( (self.p+1, self.p))
		for j in range(self.p + 1):
			ej = np.zeros(self.p + 1)
			ej[j] = 1.
			self.Dmat[j,:] = self.der(ej)

	def V(self, Y):
		M = Y.shape[0]
		V_coordinate = [self.vander(Y[:,k], self.p) for k in range(self.n)]
		
		V = np.ones((M, len(self.indices)), dtype = Y.dtype)
		
		for j, alpha in enumerate(self.indices):
			for k in range(self.n):
				V[:,j] *= V_coordinate[k][:,alpha[k]]
		return V


	def VC(self, Y, C):
		""" Compute the product V(Y) x """
		M = Y.shape[0]
		assert len(self.indices) == C.shape[0]

		if len(C.shape) == 2:
			oneD = False
		else:
			C = C.reshape(-1,1)
			oneD = True

		V_coordinate = [self.vander(Y[:,k], self.p) for k in range(self.n)]
		out = np.zeros((M, C.shape[1]))	
		for j, alpha in enumerate(self.indices):

			# If we have a non-zero coefficient
			if np.max(np.abs(C[j,:])) > 0.:
				col = np.ones(M)
				for ell in range(self.n):
					col *= V_coordinate[ell][:,alpha[ell]]

				for k in range(C.shape[1]):
					out[:,k] += C[j,k]*col
		if oneD:
			out = out.flatten()
		return out

	def DV(self, Y):
		M = Y.shape[0]
		V_coordinate = [self.vander(Y[:,k], self.p) for k in range(self.n)]
		
		mi = MultiIndex(self.n, self.p)
		N = len(mi)
		DV = np.ones((M, N, self.n), dtype = Y.dtype)

		for k in range(self.n):
			for j, alpha in enumerate(MultiIndex(self.n, self.p)):
				for q in range(self.n):
					if q == k:
						DV[:,j,k] *= np.dot(V_coordinate[q][:,0:-1], self.Dmat[alpha[q],:])
					else:
						DV[:,j,k] *= V_coordinate[q][:,alpha[q]]

		return DV


class MonomialTensorBasis(TensorBasis):
	pass

class LegendreTensorBasis(TensorBasis):
	def __init__(self, n, p):
		self.n = n
		self.p = p
		self.vander = legvander
		self.der = legder
		self.indices = index_set(p, n).astype(int)
		self.build_Dmat()

class ChebyshevTensorBasis(TensorBasis):
	def __init__(self, n, p):
		self.n = n
		self.p = p
		self.vander = chebvander
		self.der = chebder
		self.indices = index_set(p, n).astype(int)
		self.build_Dmat()

class LaguerreTensorBasis(TensorBasis):
	def __init__(self, n, p):
		self.n = n
		self.p = p
		self.vander = lagvander
		self.der = lagder
		self.indices = index_set(p,n ).astype(int)
		self.build_Dmat()

class HermiteTensorBasis(TensorBasis):
	def __init__(self, n, p):
		self.n = n
		self.p = p
		self.vander = hermvander
		self.der = hermder
		self.indices = index_set(p, n).astype(int)
		self.build_Dmat()


# Setup Joblib cache
from joblib import Memory
import os
try:
	os.mkdir('.cache')
except:
	pass
memory = Memory(cachedir = '.cache', verbose = 1) 

@memory.cache
def build_orthogonal_basis(n, p):
	# Build a basis from total degree monomials
	monomial_basis = []
	x = [Symbol('x%d' % k) for k in range(1,n+1)]
	for alpha in MultiIndex(n, p):
		term = 1
		for j in range(n):
			term *= x[j]**alpha[j]
		monomial_basis.append(term)
	
	# Now build the corresponding mass matrix
	M = zeros(len(monomial_basis), len(monomial_basis))
	for i, psi1 in enumerate(monomial_basis):
		for j, psi2 in enumerate(monomial_basis):
			if i <= j:
				out = psi1*psi2
				for k in range(n):
					out = integrate(out, (x[k], -1,1))
				M[i,j] = out
				M[j,i] = out
	
	R = M.cholesky().inv()
	
	# Now build our orthogonal basis
	basis_terms = []
	basis = []
	for i in range(len(monomial_basis)):
		term = 0
		for j, psi in enumerate(monomial_basis):
			term += R[i,j]*psi

		basis.append(term)
		# Make the sparse version
		term = Poly(term, x)
		term = [ (alpha, float(term.coeff_monomial(alpha)) ) for alpha in term.monoms()]
		basis_terms.append(term)
		
	# Now build the derivatives
	basis_terms_der = []
	for i in range(n):
		basis_terms_der_curr = []
		for psi in basis:
			# Make the sparse version
			term = Poly(diff(psi, x[i]), x)
			term = [ (alpha, float(term.coeff_monomial(alpha)) ) for alpha in term.monoms()]
			basis_terms_der_curr.append(term)

			basis_terms_der.append(basis_terms_der_curr)
	
	return basis_terms, basis_terms_der


class OrthogonalBasis(Basis):
	"""
	Parameters
	----------
	n: int
		polynomials on R^n
	p: int
		of total degree p
	"""
	def __init__(self, n, p, basis_terms = None, basis_terms_der = None):
		self.n = n
		self.p = p
		self.N = len(MultiIndex(n, p))	
		if basis_terms == None or basis_terms_der == None:
			self.basis_terms, self.basis_terms_der = build_orthogonal_basis(n, p) 

	def V(self, Y):
		""" Build a generalized multivariate Vandermonde matrix for this basis

		"""
		assert Y.shape[1] == self.n
		V_coordinate = [polyvander(Y[:,k], self.p) for k in range(self.n)]
		V = np.zeros((Y.shape[0], self.N))

		for j, terms in enumerate(self.basis_terms):
			for alpha, coeff in terms:
				# determine the coefficients on the monomial polynomial
				# Compute the product of the 
				V_col = np.ones(V.shape[0])
				for k in range(0, self.n):
					V_col *= V_coordinate[k][:,alpha[k]]
				V[:,j] += coeff * V_col
		return V


	def DV(self, Y):
		""" Build a generalized multivariate Vandermonde matrix for this basis

		"""
		M = Y.shape[0]
		assert Y.shape[1] == self.n
		V_coordinate = [polyvander(Y[:,k], self.p) for k in range(self.n)]
		DV = np.zeros((M, self.N, self.n))

		for k in range(self.n):
			for j, terms in enumerate(self.basis_terms_der[k]):
				for alpha, coeff in terms:
					# determine the coefficients on the monomial polynomial
					# Compute the product of the 
					V_col = np.ones(M)
					for i in range(self.n):
						V_col *= V_coordinate[i][:,alpha[i]]
					DV[:,j,k] += coeff * V_col
		return DV

def test_V(basis = None):
	if basis is None:
		basis = OrthogonalBasis(2,5)

	Y = np.random.uniform(-1,1, size = (10,basis.n))
	
	V = basis.V(Y)
	V2 = np.zeros(V.shape)

	for j in range(len(basis.basis)):
		psi = lambdify(basis.x, basis.basis[j], 'numpy')
		V2[:,j] = psi(*[Y[:,k] for k in range(Y.shape[1])])

	err = np.max(np.abs(V - V2))
	print "Vandermonde matrix formation error", err
	assert err < 1e-7

def residual(U, X, fX, basis, **kwargs):
	V = build_V(U, X, basis, **kwargs)
	c = lstsq(V, fX)	
	r = fX - np.dot(V, c)
	return r	

def build_V(U, X, basis, scale = True, UX = None):
	"""
		basis : ['monomial', 'legendre']
			If 'monomial', build V in the monomial basis
	"""

	M, m = X.shape
	if len(U.shape) == 1:
		U = U.reshape(m, -1)
	m, n = U.shape
	
	if UX is not None:
		Y = UX
	else:
		Y = np.dot(U.T, X.T).T
	
	if scale is True:
		if isinstance(basis, HermiteTensorBasis):
			mean = np.mean(Y, axis = 0)
			std = np.std(Y, axis = 0)
			# In numpy, 'hermite' is physicist Hermite polynomials
			# so we scale by 1/sqrt(2) to convert to the 'statisticians' Hermite 
			# polynomials which are orthogonal with respect to the standard normal
			Y = (Y - mean[None,:])/std[None,:]/np.sqrt(2)
		else:
			lb = np.min(Y, axis = 0)
			ub = np.max(Y, axis = 0)
			Y = 2*(Y-lb[None,:])/(ub[None,:] - lb[None,:]) - 1

	V = basis.V(Y)
	return V

def build_J(U, X, fX, basis, scale = True):
	"""

	Parameters
	----------
	c: np.array
		polynomial coefficients V^+fX
	"""
	M, m = X.shape
	if len(U.shape) == 1:
		U = U.reshape(m, -1)

	m, n = U.shape
	
	Y = np.dot(U.T, X.T).T
	
	if scale is True:
		if isinstance(basis, HermiteTensorBasis):
			mean = np.mean(Y, axis = 0)
			std = np.std(Y, axis = 0)
			# In numpy, 'hermite' is physicist Hermite polynomials
			# so we scale by 1/sqrt(2) to convert to the 'statisticians' Hermite 
			# polynomials which are orthogonal with respect to the standard normal
			Y = (Y - mean[None,:])/std[None,:]/np.sqrt(2)
			d_scale = 1./std
		else:
			lb = np.min(Y, axis = 0)
			ub = np.max(Y, axis = 0)
			Y = 2*(Y-lb[None,:])/(ub[None,:] - lb[None,:]) - 1
			d_scale = 2./(ub - lb)
	
	else:
		d_scale = np.ones(n)

	V = basis.V(Y)

	c = lstsq(V, fX)	
	r = fX - np.dot(V, c)

	DV = basis.DV(Y)

	# We precompute the SVD to have access to P_V^perp and V^-
	# via matrix multiplication instead of linear solves 
	Y, s, ZT = svd(V, full_matrices = False) 
	
	N = V.shape[1]
	J1 = np.zeros((M,m,n))
	J2 = np.zeros((N,m,n))

	for ell in range(n):
		for k in range(m):
			DVDU_k = X[:,k,None]*DV[:,:,ell]*d_scale[ell]
			
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


def test_residual(basis, **kwargs):
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
	r = residual(U, X, fX, basis, **kwargs)  
	assert np.all(np.isclose(r, 0))


def test_jacobian(M = 100, m = 5, basis = None, **kwargs):
	""" Test the Jacobian using finite differences
	"""
	#np.random.seed(0)

	def f(x):
		w = np.ones(x.shape)
		w /= np.linalg.norm(w)
		w2 = np.zeros(x.shape)
		w2[0] = 1
		return np.dot(x, w)**3 + np.dot(x, w2)**2 + np.dot(x,w)*np.dot(x, w2) + 10.

	if basis is None:
		n = 2
		p = 5
		basis = OrthogonalBasis(n, p)
	else:
		n = basis.n
		p = basis.p

	# Generate samples of function
	X = np.random.uniform(size = (M, m))
	fX = np.array([f(x) for x in X])

	U = np.random.randn(m,n)
	U = orth(U)
	J = build_J(U, X, fX, basis, **kwargs)

	# Finite difference approximation of the Jacobian
	h = 1e-6	
	J_est = np.zeros(J.shape)
	for k, ell in product(range(U.shape[0]), range(U.shape[1])):
		dU = np.zeros(U.shape)
		dU[k, ell] = h
		J_est[:, k, ell] = (residual(U + dU, X, fX, basis, **kwargs) - residual(U - dU, X, fX, basis, **kwargs))/(2*h)
	
	print J[0,0:5,:]/J_est[0,0:5,:]


	print "Finite difference error", np.max(np.abs(J - J_est))
	UX = np.dot(U.T, X.T)
	lb = np.min(UX, axis = 1)
	ub = np.max(UX, axis = 1)
	assert np.all(np.isclose(J, J_est))

def grassmann_gauss_newton(U0, X, fX, basis, disp = False, 
	xtol = 1e-7, ftol = 1e-7, gtol = 1e-10, beta = 1e-8, shrink = 0.5, maxiter = 100, reorth = False,
	step0 = 1., history = False, gauss_newton = True, rtol = 0, scale = True):
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
	scale: boolean
		If true, scale the projected inputs U^T X onto [-1,1]
	"""
	U = np.copy(U0)
	n, m = U.shape
	if m >= 1:
		U = orth(U)

	N, n2 = X.shape
	assert n == n2, "shapes of the subspace and X must match"
	degree = basis.p
	
	if (degree == 1 and m > 1): # "degree 1 polynomial does not permit a subspace of greater than one dimension"
		raise UnderdeterminedException

	if len(MultiIndex(m, degree)) + n*m >= N:
		raise UnderdeterminedException


	V = build_V(U, X, basis, scale = scale) 	# construct the generalized Vandermonde matrix
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
		J = build_J(U, X, fX, basis, scale = scale)

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
		maxiter2 = 50
		for it2 in range(maxiter2):
			# Compute new estimate
			U_new = np.dot(UZ, np.diag(np.cos(s*t))) + np.dot(Y, np.diag(np.sin(s*t)))
			# Enforce orthogonality more strictly than the above expression
			U_new = orth(U_new)
			
			# Compute the new residual
			UX_new = np.dot(U_new.T, X.T)
			V_new = build_V(U_new, X, basis, scale = scale)
			c_new = lstsq(V_new, fX)
			r_new = fX - np.dot(V_new, c_new)	
			norm_r_new = float(norm(r_new))
			#print "decrease", norm_r - norm_r_new, norm_r_new/norm_r, "alpha", alpha, "beta", beta, "t", t, "grad %1.4e %1.4e" % (np.max(np.abs(G)),np.min(np.abs(G)))
			if norm_r_new <= norm_r + alpha * beta * t or (norm_r_new < norm_r and (norm_r_new/norm_r) < 0.9): 
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
	def __init__(self, degree = None, subspace_dimension = None, n_init = 1, basis = 'legendre', scale = True, **kwargs):
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
		if isinstance(basis, basestring):
			if basis == 'monomial':
				basis = MonomialTensorBasis(subspace_dimension, degree)
			elif basis == 'legendre':
				basis = LegendreTensorBasis(subspace_dimension, degree)
			elif basis == 'hermite':
				basis = HermiteTensorBasis(subspace_dimension, degree)
			elif basis == 'laguerre':
				basis = LaguerreTensorBasis(subspace_dimension, degree)
			elif basis == 'orthogonal':
				basis = OrthogonalBasis(subspace_dimension, degree)
		elif isinstance(basis, Basis):
			degree = basis.p
		else:
			raise NotImplementedError('Basis type not understood')


		if subspace_dimension is 0 and degree is None:
			degree = 0
		if subspace_dimension is 1 and degree is None:
			degree = 1
		if degree is 0 and subspace_dimension is None:
			subspace_dimension = 0
		if degree is 1 and subspace_dimension is None:
			subspace_dimension = 1

		if degree is 1 and subspace_dimension != 1:
			raise IllposedException('Affine linear functions intrinsically only have a 1 dimensional subspace')
		if degree is 0 and subspace_dimension > 0:
			raise IllposedException('The constant function does not have a subspace associated with it')
		if subspace_dimension is 0 and degree > 1:
			raise IllposedException('Zero-dimensional subspaces cannot have a polynomial term associated with them')

		self.degree = degree
		self.subspace_dimension = subspace_dimension
		self.kwargs = kwargs
		self.n_init = n_init
		self.basis = basis
		self.scale = scale

	def fit(self, X, y, U_fixed = None):
		""" Build ridge function approximation
		"""
		
		self.X = np.copy(X)
		self.y = np.copy(y)

		# If we have been provided with a fixed U
		if U_fixed is not None:
			self.U = orth(U_fixed)
			V = build_V(self.U, X, self.basis,  scale = self.scale)
			self.c = lstsq(V,y)
			return

		# Special case of fitting a constant
		if self.subspace_dimension == 0 and self.degree == 0:
			self.U = np.zeros((X.shape[1], 0))
			self.c = np.linalg.lstsq(build_V(self.U, X, self.basis, scale = self.scale), y)[0]
			return
	
		# Special case of fitting an affine linear fit	
		if self.degree == 1 and self.subspace_dimension == 1:
			# Solve the linear least squares problem
			XX = np.hstack([X, np.ones((X.shape[0],1))])
			b = np.linalg.lstsq(XX, y)[0]
			self.U = b[0:-1].reshape(-1,1)
			U_norm = np.linalg.norm(self.U, 2)
			self.U /= U_norm
			self.c = np.array([b[-1], U_norm])
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
	
	
		self.U, self.c = grassmann_gauss_newton(U0, X, y, self.basis, **kwargs)

		# Try other initializations
		if self.n_init > 1:
			res_norm_best = self.score(X, y)
			U_best, c_best = np.copy(self.U), np.copy(self.c)
		else:
			return

		for it in range(1, self.n_init):
			U0 = orth(np.random.randn(X.shape[1], self.subspace_dimension))
			self.U, self.c = grassmann_gauss_newton(U0, X, y, self.basis, **kwargs)
			res_norm_cur = self.score(X, y)
			if res_norm_cur < res_norm_best:
				U_best, c_best = self.U, self.c
		
		self.U, self.c = U_best, c_best
	
	def refine(self, n_init = 1, **kwargs):
		"""Improve the current estimate
		"""
		U_best, c_best = np.copy(self.U), np.copy(self.c)
		res_norm_best = self.score(self.X, self.y)
		for it in range(n_init):
			U0 = orth(np.random.randn(self.X.shape[1], self.subspace_dimension))
			self.U, self.c = grassmann_gauss_newton(U0, self.X, self.y, self.basis, **kwargs)
			res_norm_cur = self.score(self.X, self.y)
			if res_norm_cur < res_norm_best:
				U_best, c_best = self.U, self.c

		self.U, self.c = U_best, c_best
	
	def predict(self, X):
		Ynew = np.dot(self.U.T, X.T).T	
		
		if self.scale is True:
			Y = np.dot(self.U.T, self.X.T).T
			if isinstance(self.basis, HermiteTensorBasis):
				mean = np.mean(Y, axis = 0)
				std = np.std(Y, axis = 0)
				# In numpy, 'hermite' is physicist Hermite polynomials
				# so we scale by 1/sqrt(2) to convert to the 'statisticians' Hermite 
				# polynomials which are orthogonal with respect to the standard normal
				Ynew = (Ynew - mean[None,:])/std[None,:]/np.sqrt(2)
			else:
				lb = np.min(Y, axis = 0)
				ub = np.max(Y, axis = 0)
				Ynew = 2*(Ynew-lb[None,:])/(ub[None,:] - lb[None,:]) - 1
			
		V = self.basis.V(Ynew) 
		return np.dot(V, self.c)

	def predict_ridge(self, Y):
		V = self.basis.V(Y)
		return np.dot(V, self.c)
	
	def score(self, X = None, y = None, norm = False):
		if X is None and y is None:
			X = self.X
			y = self.y
		if X is None or y is None:
			raise RuntimeError('Please provide both X and y')

		diff = np.linalg.norm(self.predict(X) - y, 2)
		if norm:
			return diff/np.linalg.norm(y,2)
		else:
			return diff

	def plot(self, axes = None):
		from matplotlib import pyplot as plt
		if axes is None:
			fig, axes = plt.subplots(figsize = (6,6))

		if self.subspace_dimension == 1:
			Y = np.dot(self.U.T, self.X.T).flatten()
			lb = np.min(Y)
			ub = np.max(Y)
			
			axes.plot(Y, self.y, 'k.', markersize = 6)
			xx = np.linspace(lb, ub, 100)
			XX = np.array([self.U.flatten()*x for x in xx])
			axes.plot(xx, self.predict(XX), 'r-', linewidth = 2)

		if self.subspace_dimension == 2:
			Y = np.dot(self.U.T, self.X.T).T
			# Construct grid
			x = np.linspace(np.min(Y[:,0]), np.max(Y[:,0]), 100)	
			y = np.linspace(np.min(Y[:,1]), np.max(Y[:,1]), 100)
			xx, yy = np.meshgrid(x, y)
			# Sample the ridge function
			UXX = np.vstack([xx.flatten(), yy.flatten()])
			XX = np.dot(self.U, UXX).T
			YY = self.predict(XX)
			YY = np.reshape(YY, xx.shape)
			
			axes.contour(xx, yy, YY, 20)
			
			# Plot points
			axes.scatter(Y[:,0], Y[:,1], c = self.y, s = 6)
		return axes

	def box_domain(self):
		""" Return the lower and upper bounds on the active domain

		This only depends on the set of points given as input, so we don't extrapolate too much.
		A convex hull provides a tighter description in higher dimensional space. 
		"""
		UX = np.dot(self.U.T, self.X.T).T
		lb = np.min(UX, axis = 0)
		ub = np.max(UX, axis = 0)
		return [lb, ub] 




if __name__ == '__main__':
	import sys
	# Tests to ensure the residual and Jacobian are calculated correctly
	test_residual()
	test_jacobian(basis = 'legendre', scale = True)


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


