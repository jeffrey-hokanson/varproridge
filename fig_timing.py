import numpy as np
from itertools import product
from scipy.linalg import orth
from time import time

from joblib import Memory
from tqdm import tqdm
from ridge import *
from ridge_paul import RidgeAlternating

memory = Memory(cachedir = '.fig_timing', verbose = 0)

@memory.cache
def generate_timings(degree = 3, subspace_dimension = 1, seed = 0, m = 10, M = 1000, which = 'gn', rtol = 1e-5, **kwargs):
	# First define a function of specified degree on a subspace of dimension
	def f(x):
		w = np.ones(x.shape)
		val = np.dot(x.T, w)**degree
		for i in range(subspace_dimension - 1):
			w = np.zeros(x.shape)
			w[i] = 1.
			val += np.dot(x, w)**(degree-1)
		return val + 1
	
	np.random.seed(0)
	X = np.random.uniform(size = (M,m))
	fX = np.array([f(x) for x in X])
	
	Utrue = np.zeros((m, subspace_dimension))
	Utrue[:,0] = np.ones(m)
	for i in range(subspace_dimension - 1):
		Utrue[i,i+1] = 1.
	Utrue = orth(Utrue)
	
	np.random.seed(seed)
	U0 = orth(np.random.randn(m,subspace_dimension))
	if which == 'gn':
		start_time = time()
		basis = LegendreTensorBasis(subspace_dimension, degree)
		U, c = grassmann_gauss_newton(U0, X, fX, basis, xtol = -1, ftol = -1, gtol = -1, rtol = rtol,disp = False, **kwargs)
		duration = time() - start_time
		subspace_error = np.min(np.linalg.svd(np.dot(orth(U).T, Utrue), compute_uv = False))
		return duration, subspace_error
		#print fX - np.dot(build_V(U, X, degree), c)
		#return t
	else:
		start_time = time()
		# Paul's code defines the residual as 0.5*np.linalg.norm(f-g)**2
		# so we alter the convergence tolerance to match
		U = RidgeAlternating(X, fX, U0, degree = degree, tol = 0.5*rtol**2, **kwargs)
		duration = time() - start_time
		subspace_error = np.min(np.linalg.svd(np.dot(orth(U).T, Utrue), compute_uv = False))
		return duration, subspace_error


def generate_timing(name, which = 'gn', n_trials = 10, max_degree = 5, max_subspace = 5, iters = [10], **kwargs):
	
	timing = np.nan*np.ones((max_degree + 1,max_subspace+1, n_trials))
	err = np.nan*np.ones((max_degree + 1,max_subspace+1, n_trials))
	for degree in range(2,max_degree+1):
		for subspace_dimension in range(1,max_subspace + 1):
			for trial in range(n_trials):
				if which != 'gn':
					timing[degree, subspace_dimension, trial], err[degree, subspace_dimension, trial] = generate_timings(degree, subspace_dimension, seed = trial, which = which, **kwargs)
					print 'degree %d, dimension %d, trial %d, time %g, err %g' % (degree, subspace_dimension, trial, timing[degree, subspace_dimension, trial], err[degree, subspace_dimension, trial])
				else:
					for inner_iter in iters:
						t, e = generate_timings(degree, subspace_dimension, seed = trial, which = which, **kwargs)
						timing[degree, subspace_dimension, trial] = np.nanmin([t, timing[degree, subspace_dimension, trial]])
					print 'degree %d, dimension %d, trial %d, time %g, err %g' % (degree, subspace_dimension, trial, timing[degree, subspace_dimension, trial], e)

	from pgf import PGF
	pgf = PGF()
	pgf.add('degree', np.arange(2,max_degree+1))
	for dim in range(1,max_subspace+1):
		pgf.add('m%d' % dim, [np.median(timing[d, dim,:]) for d in range(2,max_degree+1)])
	pgf.write(name)


if __name__ == '__main__':
	n_trials = 10
	max_degree = 5
	max_subspace = 5
	generate_timing('fig_timing_gn_m10.dat', which = 'gn', max_degree = max_degree, max_subspace = max_subspace, n_trials = n_trials, m = 10)
	#generate_timing('fig_timing_gn_m100.dat', which = 'gn', max_degree = max_degree, max_subspace = max_subspace, n_trials = n_trials, m = 100)
	print "Alt 1"
	generate_timing('fig_timing_alt_m10_1.dat', which = 'alt', max_degree = max_degree, max_subspace = max_subspace,
		 n_trials = n_trials, iters = [1], m=10)
	#generate_timing('fig_timing_alt_m100_1.dat', which = 'alt', max_degree = max_degree, max_subspace = max_subspace,
	#	 n_trials = n_trials, iters = [1], m=100)
	print "Alt 10"
	generate_timing('fig_timing_alt_10.dat', which = 'alt', max_degree = max_degree, max_subspace = max_subspace, n_trials = n_trials, iters = [10], m =10)
	print "Alt 100"
	generate_timing('fig_timing_alt_100.dat', which = 'alt', max_degree = max_degree, max_subspace = max_subspace, n_trials = n_trials, iters = [100], m = 10)
	generate_timing('fig_timing_alt_best.dat', which = 'alt', max_degree = max_degree, max_subspace = max_subspace, n_trials = n_trials, iters = [1,10,100], m = 10)
