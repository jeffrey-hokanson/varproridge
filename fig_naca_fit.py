import numpy as np
from joblib import Memory
from tqdm import tqdm
from ridge import PolynomialRidgeApproximation
from sparse import SparsePolynomial 
from gaussian_process import GaussianProcess
from pgf import PGF
from global_polynomial import GlobalPolynomial

X = np.loadtxt('naca_lhs.input')
fX = np.loadtxt('naca_lhs.output')
f_drag = fX[:,0]
f_lift = fX[:,1]



memory = Memory(cachedir = '.fig_naca', verbose = 0)

@memory.cache
def quad_fit(M = 100, seed = 0, obj = 'lift'):
	gp = GlobalPolynomial(degree = 2)

	np.random.seed(seed)
	I = np.random.permutation(X.shape[0])[:M]
	if obj == 'lift':
		gp.fit(X[I,:], f_lift[I])
	elif obj == 'drag':
		gp.fit(X[I,:], f_drag[I])
	return gp

@memory.cache
def pra_fit(M = 107, seed = 0,  degree = 5, n_init = 1, subspace_dimension = 1, obj = 'lift'):
	pra = PolynomialRidgeApproximation(subspace_dimension = subspace_dimension, degree = degree, n_init = n_init)
	
	# Make data
	np.random.seed(seed)
	I = np.random.permutation(X.shape[0])[:M]
	if obj == 'lift':
		pra.fit(X[I,:], f_lift[I])
	elif obj == 'drag':
		pra.fit(X[I,:], f_drag[I])
	return pra 
	
@memory.cache
def sp_fit(M = 107, seed = 0, degree = 3, obj = 'lift', **kwargs):
	sp = SparsePolynomial(degree, **kwargs)
	np.random.seed(seed)
	perm = np.random.permutation(X.shape[0])
	I = perm[:M]
	
	if obj == 'lift':
		sp.fit(X[I,:], f_lift[I])
	elif obj == 'drag':
		sp.fit(X[I,:], f_drag[I])
	return sp


@memory.cache
def gp_fit(M = 108, seed = 0, obj  = 'lift'):
	# Make data
	np.random.seed(seed)
	I = np.random.permutation(X.shape[0])[:M]
	gp = GaussianProcess()	

	if obj == 'lift':
		gp.fit(X[I,:], f_lift[I])
	elif obj == 'drag':
		gp.fit(X[I,:], f_drag[I])
	return gp


def generate_data(name, func, Ms, Nit, obj = 'lift',**kwargs):
	perc = np.zeros((len(Ms), 3))
	for j, M in enumerate(Ms):
		print "starting", M
		scores = []
		for seed in tqdm(range(Nit), desc = 'seed'):
			fit = func(M = M, seed = seed, obj = obj, **kwargs)
			if obj == 'lift':
				scores.append(fit.score(X, f_lift)/np.linalg.norm(f_lift))
			elif obj == 'drag':
				scores.append(fit.score(X, f_drag)/np.linalg.norm(f_drag))
				
			perc[j,:] = np.percentile(scores, [25, 50, 75])

			# Write out intermedate results
			pgf = PGF()
			pgf.add('M', Ms)
			pgf.add('p25', perc[:,0])
			pgf.add('p50', perc[:,1])
			pgf.add('p75', perc[:,2])
			pgf.write('fig_naca_%s_%s.dat' % (obj, name ))

Nit = 100

if False:
	print "Line"
	Ms = [25,50,100, 200, 500, 1000, 2000, 5000, 10000]
	generate_data('line', pra_fit, Ms, Nit, degree = 1, subspace_dimension = 1, obj = 'lift')
	generate_data('line', pra_fit, Ms, Nit, degree = 1, subspace_dimension = 1, obj = 'drag')


if False:
	print "pra 1,5"
	generate_data('pra_1_5', pra_fit, Ms, Nit, degree = 5, subspace_dimension = 1, obj = 'lift', n_init = 10)
	generate_data('pra_1_5', pra_fit, Ms, Nit, degree = 5, subspace_dimension = 1, obj = 'drag', n_init = 10)

if False:
	Ms = [70,100, 200, 500, 1000, 2000, 5000, 10000]
	print "pra 2,5"
	generate_data('pra_2_5', pra_fit, Ms, Nit, degree = 5, subspace_dimension = 2, obj = 'lift', n_init = 10)
	generate_data('pra_2_5', pra_fit, Ms, Nit, degree = 5, subspace_dimension = 2, obj = 'drag', n_init = 10)

if False:
	Ms = [150, 200, 500, 1000, 2000, 5000, 10000]
	print "pra 3,5"
	generate_data('pra_3_5', pra_fit, Ms, Nit, degree = 5, subspace_dimension = 3, obj = 'lift', n_init = 10)
	generate_data('pra_3_5', pra_fit, Ms, Nit, degree = 5, subspace_dimension = 3, obj = 'drag', n_init = 10)

if False:
	Nit = 10
	Ms = [500, 1000, 2000, 5000, 10000]
	print "pra 4,5"
	generate_data('pra_4_5', pra_fit, Ms, Nit, degree = 5, subspace_dimension = 4, obj = 'lift', n_init = 10)
	generate_data('pra_4_5', pra_fit, Ms, Nit, degree = 5, subspace_dimension = 4, obj = 'drag', n_init = 10)

if False:
	Ms = [500, 1000, 2000, 5000, 10000]
	print "pra 5,5"
	generate_data('pra_5_5', pra_fit, Ms, Nit, degree = 5, subspace_dimension = 5, obj = 'lift', n_init = 10)
	generate_data('pra_5_5', pra_fit, Ms, Nit, degree = 5, subspace_dimension = 5, obj = 'drag', n_init = 10)

if True:
	Ms = [200, 500, 1000, 2000, 5000, 10000]
	print "quadratic"
	generate_data('quad', quad_fit, Ms, Nit, obj = 'lift')
	generate_data('quad', quad_fit, Ms, Nit, obj = 'drag')

if False:
	Ms = [20,50, 100, 150, 200, 500, 1000, 2000, 5000, 10000]
	print "Gaussian Process"
	generate_data('gp', gp_fit, Ms, Nit, obj = 'lift')
	generate_data('gp', gp_fit, Ms, Nit, obj = 'drag')

if False:
	Ms = [20,50, 100, 150, 200, 500, 1000, 2000, 5000, 10000]
	print "Sparse"
	generate_data('sp', sp_fit, Ms, Nit, obj = 'lift')
	generate_data('sp', sp_fit, Ms, Nit, obj = 'drag')
