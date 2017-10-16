import numpy as np
from pde_model import PDESolver
from tqdm import tqdm

N = int(1e6)
np.random.seed(0)
X = np.random.randn(N,100)

test = True
if test:
	np.random.seed(42)

fX = np.zeros(N)
gradX = np.zeros((N, 100))
for i in tqdm(range(N)):
	pde_long = PDESolver('long')
	fX[i] = pde_long.qoi(X[i,:])
	#gradX[i,:] = pde_long.grad_qoi(X[i,:])

if test:
	np.savetxt('pde_long_test.input', X)
	np.savetxt('pde_long_test.output', fX)
else:
	np.savetxt('pde_long.input', X)
	np.savetxt('pde_long.output', fX)
	#np.savetxt('pde_long_grad.output', gradX)
