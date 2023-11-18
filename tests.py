import unittest
from src.data_structures.Instance import Instance
from src.solvers.collection import SolverCollection,SolverConfig

class TestStringMethods(unittest.TestCase):

    def test_generator(self):
        """Can i generate an instance?"""
        Instance.generate(n_items = 10, gamma = 0)


    def test_gurobi_solver(self):
        """Can i use gurobi?"""
        instance = Instance.generate(n_items = 10, gamma = 0)
        SolverCollection.gurobi(instance,solver_config=SolverConfig.optimal())


if __name__ == '__main__':
    unittest.main()