import numpy as np
from ridge import PolynomialRidgeApproximation

from pgf import PGF


X = np.loadtxt('naca_lhs.input')
fX = np.loadtxt('naca_lhs.output')
f_drag = fX[:,0]
f_lift = fX[:,1]

for fX, name in zip([f_drag, f_lift], ['drag', 'lift']):
	pra = PolynomialRidgeApproximation(subspace_dimension = 1, degree = 5, n_init =10)
	pra.fit(X, fX)

	UX = np.dot(pra.U.T, X.T).flatten()

	I = np.argsort(UX).flatten()
	y = pra.predict(X)

	pgf = PGF()
	pgf.add('UX', UX[I][::10])
	pgf.add('y', y[I][::10])
	pgf.add('fX', fX[I][::10])
	pgf.write('fig_naca_%s_ridge.dat' % name)


	pgf = PGF()
	pgf.add('i', np.arange(18))
	pgf.add('Ui', pra.U.flatten())

	pgf.write('fig_naca_%s_ridge_U.dat' % name)
