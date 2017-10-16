import numpy as np
from ridge import PolynomialRidgeApproximation

from pgf import PGF


X = np.loadtxt('pde_long.input')
fX = np.loadtxt('pde_long.output')
print "data loaded"



X = X[:5000]
fX = fX[:5000]

pra = PolynomialRidgeApproximation(subspace_dimension = 1, degree = 7, n_init =10)
pra.fit(X, fX)

UX = np.dot(pra.U.T, X.T).flatten()

I = np.argsort(UX).flatten()
print I
y = pra.predict(X)

pgf = PGF()
pgf.add('UX', UX[I][::10])
pgf.add('y', y[I][::10])
pgf.add('fX', fX[I][::10])
pgf.write('fig_pde_ridge.dat')


pgf = PGF()
pgf.add('i', np.arange(100))
pgf.add('Ui', pra.U.flatten())

pgf.write('fig_pde_ridge_U.dat')
