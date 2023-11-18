import numpy as np
from src.data_structures.instance import Instance

from src.solvers.collection import SolverCollection
instance = Instance.generate(100,10)
np.random.seed(100)
sol = SolverCollection.gurobi_optimal(instance)
of = instance.fast_evaluate(sol.sol)
print(sol.o,of)