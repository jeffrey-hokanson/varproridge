import numpy as np
from joblib import Memory
from tqdm import tqdm
from ridge import PolynomialRidgeApproximation
from sparse import SparsePolynomial 
from gaussian_process import GaussianProcess
from pgf import PGF



X = np.loadtxt('pde_long.input')
fX = np.loadtxt('pde_long.output')
X_test = np.loadtxt('pde_long_test.input')
fX_test = np.loadtxt('pde_long_test.output')
print "data loaded"

memory = Memory(cachedir = '.fig_pde', verbose = 0)


@memory.cache
def pra_fit(M = 107, seed = 0,  degree = 5, n_init = 1, subspace_dimension = 1):
	pra = PolynomialRidgeApproximation(subspace_dimension = subspace_dimension, degree = degree, n_init = n_init)
	
	# Make data
	np.random.seed(seed)
	I= np.random.permutation(X.shape[0])[:M]
	pra.fit(X[I,:], fX[I])
	return pra 
	
@memory.cache
def sp_fit(M = 107, seed = 0, degree = 3, crossvalidate = False, **kwargs):
	sp = SparsePolynomial(degree, **kwargs)
	np.random.seed(seed)
	perm = np.random.permutation(X.shape[0])
	I = perm[:M]
	if crossvalidate:
		I2 = perm[M:2*M]
		sp.fit(X[I,:], fX[I], X_test = X[I2,:], fX_test = fX[I2])
	else:
		sp.fit(X[I,:], fX[I])
	return sp


@memory.cache
def gp_fit(M = 108, seed = 0):
	# Make data
	np.random.seed(seed)
	I = np.random.permutation(X.shape[0])[:M]
	gp = GaussianProcess()	
	gp.fit(X[I,:], fX[I])
	return gp



def generate_data(name, func, Ms, Nit,**kwargs):
	perc = np.zeros((len(Ms), 3))
	for j, M in enumerate(Ms):
		print "starting", M
		scores = []
		for seed in tqdm(range(Nit), desc = 'seed'):
			fit = func(M = M, seed = seed, **kwargs)
			score = 0
			for i in range(10):
				score += fit.score(X_test[int(1e5)*i:int(1e5)*(i+1)], fX_test[int(1e5)*i:int(1e5)*(i+1)] )**2
			scores.append(float(np.sqrt(score)/np.linalg.norm(fX_test)))
		
				
			perc[j,:] = np.percentile(scores, [25, 50, 75])

			# Write out intermedate results
			pgf = PGF()
			pgf.add('M', Ms)
			pgf.add('p25', perc[:,0])
			pgf.add('p50', perc[:,1])
			pgf.add('p75', perc[:,2])
			pgf.write('fig_pde_long_%s.dat' % name )


if __name__ == '__main__':
	Ms = [107, 120, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 5000,10000,20000,50000]
	if True:
		Nit = 10
		Ms = [107, 120, 150, 200, 500,  1000, 2000, 5000,10000, 20000, 50000]
		print "Gaussian Process"
		generate_data('gp', gp_fit, Ms, Nit)
	if False:
		Nit = 10
		Ms = [107, 120, 200, 500,  1000, 2000, 5000, 10000, 20000, 50000]
		print "Sparse"
		generate_data('sp', sp_fit, Ms, Nit, degree = 3)


	Nit = 100
	if False:
		print "Polynomial 1,2"
		generate_data('pra_1_2',pra_fit, Ms, Nit, degree = 2, n_init = 10)
		print "Polynomial 1,3"
		generate_data('pra_1_3',pra_fit, Ms, Nit, degree = 3, n_init = 10)
		print "Polynomial 1,4"
		generate_data('pra_1_4',pra_fit, Ms, Nit, degree = 4, n_init = 10)
		print "Polynomial 1,5"
		generate_data('pra_1_5',pra_fit, Ms, Nit, degree = 5, n_init = 10)
		print "Polynomial 1,7"
		Ms[0] = 110
		generate_data('pra_1_7',pra_fit, Ms, Nit, degree = 7, n_init = 10)
		print "Polynomial 2,5"
		Ms = [250, 300, 400, 500, 600,700,800,900,1000, 2000, 5000, 10000,20000,50000]
		generate_data('pra_2_5',pra_fit, Ms, Nit, degree = 5, subspace_dimension = 2, n_init = 10)

