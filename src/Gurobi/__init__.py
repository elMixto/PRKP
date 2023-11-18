from .gurobi_solver import SolverConfig, solve_polynomial_knapsack, VAR_TYPE
from src.data_structures import Instance,Solution

def gurobi(instance: Instance, solver_config: SolverConfig) -> Solution:
    o, sol, t = solve_polynomial_knapsack(instance, solver_config)
    return Solution(o, sol, t)