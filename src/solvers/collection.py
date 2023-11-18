from src.solvers.gurobi import SolverConfig, solve_polynomial_knapsack, VAR_TYPE
from src.solvers.MLHeu.functions_ml import prepare_set, fix_variables
from src.config import MLH_MODEL
from src.solvers.ga.GAHeuristic import GAHeuristic
from src.data_structures.instance import Instance
from time import time
import numpy as np
import torch
from pathlib import Path
import pickle
from numpy.typing import ArrayLike
from dataclasses import dataclass




@dataclass
class Solution:
    o: float
    sol: ArrayLike
    time: float  #Seconds

    def __repr__(self) -> str:
        return f"Sol(of:{self.o},time:{self.time})"


class SolverCollection:

    @staticmethod
    def gurobi_local(instance: Instance,
                     solver_config: SolverConfig = SolverConfig.optimal()):
        return solve_polynomial_knapsack(instance, solver_config)
    
    @staticmethod
    def gurobi(instance: Instance, solver_config: SolverConfig) -> Solution:
        o, sol, t = SolverCollection.gurobi_local(instance, solver_config)
        return Solution(o, sol, t)

    @staticmethod
    def gurobi_optimal(instance: Instance) -> Solution:
        return SolverCollection.gurobi(instance, SolverConfig.optimal())

    @staticmethod
    def gurobi_continous(instance: Instance) -> Solution:
        return SolverCollection.gurobi(instance, SolverConfig.continous())

    @staticmethod
    def baldo_GA(instance: Instance,
                 n_chromosomes: int = 70,
                 penalization: float = 0.03,
                 weight: float = 0.6) -> Solution:
        start = time()
        solution = SolverCollection.gurobi(
            instance, SolverConfig(VAR_TYPE.CONTINOUS, False, []))
        ga_solver = GAHeuristic(list(solution.sol), instance, n_chromosomes,
                                penalization, weight)
        solGA, objfun = ga_solver.run()
        solGA = list(map(lambda x: int(x), list(solGA)))
        solGA = list(solGA)
        return Solution(objfun, solGA, time() - start)

    @staticmethod
    def baldo_ML(instance: Instance) -> Solution:
        start = time()
        model_file = MLH_MODEL
        n_features = 6
        fixed_percentage = 0.85
        clf = pickle.load(open(model_file, 'rb'))
        sol_cont = SolverCollection.gurobi(
            instance, SolverConfig(VAR_TYPE.CONTINOUS, False, []))
        X = prepare_set(n_features, instance, sol_cont.sol)
        y_mlProba = clf.predict_proba(X)
        y_ml = fix_variables(instance.n_items, y_mlProba, fixed_percentage)
        discrete_config = SolverConfig(VAR_TYPE.BINARY, True, y_ml)
        final_gurobi = SolverCollection.gurobi(instance, discrete_config)
        return Solution(final_gurobi.o, final_gurobi.sol, time() - start)

    @staticmethod
    def DL2(instance: Instance):
        from src.solvers.DLHEU2 import DHEU
        from src.data_structures.features import Budget, ProfitOverBudget, LowerCostOverBudget, UpperCostOverBudget
        from src.data_structures.features import NItems, Noise, ItemBatchFeature, IsInContSol, CountPSynergiesOverBudget, CountPSynergiesOverNItems
        features: list[ItemBatchFeature] = [
            ProfitOverBudget(),
            LowerCostOverBudget(),
            UpperCostOverBudget(),
            IsInContSol(),
            Noise(),
            CountPSynergiesOverNItems(),
            CountPSynergiesOverBudget()
        ]
        heu = DHEU(features)
        heu.load(Path("/home/mixto/repositories/PRKP/models/DHEUV2.model"))
        preds = heu.evaluate(instance).view(1, -1)
        print(preds)
        y_ml = fix_variables(instance.n_items, preds.T, 0.85)
        print(y_ml)


    @staticmethod
    def ZeroReductor(instance,pred_threshold: float = 0.01) -> Solution:
        from src.solvers.ActorCritic.solver import ZeroReductor2
        from src.solvers.DLHEU2 import DHEU
        from src.data_structures.features import ProfitOverBudget,LowerCostOverBudget,UpperCostOverBudget
        from src.data_structures.features import IsInContSol,CountPSynergiesOverNItems,CountPSynergiesOverBudget
        from src.data_structures.features import GammaOverNItems,SumOfSynergiesByItemOverMaxSinergyProfit,NOverON
        model = Path("/home/mixto/repositories/PRKP/models/DHEUV2_expanded.model")
        features = [
            ProfitOverBudget(),LowerCostOverBudget(),
            UpperCostOverBudget(),IsInContSol(),
            CountPSynergiesOverNItems(),
            CountPSynergiesOverBudget(),GammaOverNItems(),
            SumOfSynergiesByItemOverMaxSinergyProfit(),
            NOverON(instance)]
        heu = DHEU(features)
        heu.load(model)
        stuck = 0
        zr = ZeroReductor2(instance)
        zr.heu = heu
        done = False
        while not done:
            zr.step(max_step=200,threshold=pred_threshold)
            old_n_items = zr.instance.n_items
            if zr.instance.n_items != old_n_items:
                stuck = 0
            else:
                stuck += 1
            if torch.min(zr.actual_pred) > pred_threshold or stuck > 2 or zr.instance.n_items < 150:
                done = True
            print(zr.instance.n_items)
        return zr.solve()

