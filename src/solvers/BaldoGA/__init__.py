from .GAHeuristic import GAHeuristic
from src.data_structures import Instance,Solution
from src.Gurobi import gurobi,SolverConfig,VAR_TYPE
from time import time

def solve(instance: Instance,n_chromosomes: int = 70,
             penalization: float = 0.03,
             weight: float = 0.6) -> Solution:
    start = time()
    solution = gurobi(instance, SolverConfig(VAR_TYPE.CONTINOUS, False, []))
    ga_solver = GAHeuristic(list(solution.sol), instance, n_chromosomes,penalization, weight)
    solGA, objfun = ga_solver.run()
    solGA = list(map(lambda x: int(x), list(solGA)))
    solGA = list(solGA)
    return Solution(objfun, solGA, time() - start)
    