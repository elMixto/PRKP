from src.data_structures import Solution,Instance
from pathlib import Path
from src.data_structures.features import ProfitOverBudget,LowerCostOverBudget,UpperCostOverBudget
from src.data_structures.features import IsInContSol,CountPSynergiesOverNItems,CountPSynergiesOverBudget
from src.data_structures.features import GammaOverNItems,SumOfSynergiesByItemOverMaxSinergyProfit,NOverON
from .DLHEU2 import DHEU
from .solver import ZeroReductor2
import torch
from time import time
from src.Gurobi import gurobi,VAR_TYPE,SolverConfig
import numpy as np
from .solver import features

def solve(instance: Instance,pred_threshold: float = 0.01, heuristic_step_proportion = None,verbose=False) -> Solution:
    model = Path(__file__).resolve().parent / "models/DHEUV2.model"
    #model = Path(__file__).resolve().parent / "models/DHEUV2.model"
    heu = DHEU(features)
    heu.load(model)
    heu.net.eval()
    stuck = 0
    zr = ZeroReductor2(instance)
    zr.heu = heu
    done = False
    if heuristic_step_proportion is None:
        heuristic_step = int(zr.instance.n_items/np.log10(zr.instance.n_items))
    else:
        heuristic_step = int(zr.instance.n_items*heuristic_step_proportion)
    
    #El tamaño de paso debe ser constante
    while not done:
        zr.step(max_step = max(heuristic_step,100) ,threshold=pred_threshold)
        if verbose:
            print(zr.instance.n_items)
            print(torch.min(zr.actual_pred))
        old_n_items = zr.instance.n_items
        if zr.instance.n_items != old_n_items:
            stuck = 0
        else:
            stuck += 1
        if torch.min(zr.actual_pred) > pred_threshold or stuck > 3:
            done = True
    return zr.solve()


def solve_once(instance: Instance) -> Solution:
    from src.solvers.BaldoML.functions_ml import fix_variables
    start = time()
    features = [ProfitOverBudget(),LowerCostOverBudget(),
        UpperCostOverBudget(),IsInContSol(),
        CountPSynergiesOverNItems(),
        CountPSynergiesOverBudget(),GammaOverNItems(),
        SumOfSynergiesByItemOverMaxSinergyProfit(),
        NOverON(instance)]
    heu = DHEU(features)
    model = Path(__file__).resolve().parent / "models/DHEUV2_expanded.model"
    heu.load(model)
    preds = heu.evaluate(instance).view(1, -1)[0]
    preds = torch.stack([1- preds,preds])
    y_ml = fix_variables(instance.n_items, preds.T, 1-0.85)
    solution = gurobi(instance,SolverConfig(VAR_TYPE.BINARY,True,y_ml))
    solution = Solution(solution.o,solution.sol,solution.time + time() - start)
    return solution

