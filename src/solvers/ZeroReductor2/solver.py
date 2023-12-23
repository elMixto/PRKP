import torch
from torch import optim
import time
from pathlib import Path
import numpy as np
from src.Gurobi.gurobi_solver import VAR_TYPE
from src.data_structures.features import *
from src.data_structures import Instance, Solution
from src.solvers.ZeroReductor2.DLHEU2 import DHEU
from src.Gurobi import SolverConfig,gurobi


features: list[ItemBatchFeature] = [
            ProfitOverBudget(),
            LowerCostOverBudget(),
            UpperCostOverBudget(),
            IsInContSol(),
            GammaOverNItems(),
            Density(),
            Noise()
            ]



class ZeroReductor2:
    def __init__(self,instance: Instance) -> None:
        self.o_instance = instance
        self.global_mask = np.full(self.o_instance.n_items, -1)
        self.instance_mask = np.arange(self.o_instance.n_items,dtype=np.uint32)
        self.instance: Instance  = self.o_instance
        self.pop_array = lambda array,index: np.concatenate([array[:index],array[index+1:]])
        self.features = features
        self.heu = DHEU(self.features)
        model = Path(__file__).resolve().parent / "models/DHEUV2.model"
        self.heu.load(model)
        self.heu.net.eval()
        #Prediccion inicial
        self.actual_pred: torch.Tensor = self.heu.evaluate(self.instance)
        #Desde aqui cacheo la solucion continua optima
        self.cached_cont_sol = self.instance.get_feature(IsInContSol())
        self.step_size = 1
        self.treshold = 0.3
        self.time = 0
        self.n_steps = 0
    
    def fix_from_pred(self,pred: torch.Tensor,max_step: int,threshold: float):
        mascara = torch.lt(pred, threshold)
        pred_indexes = torch.nonzero(mascara).squeeze()
        tensor_filtrado = pred[pred_indexes]
        if pred_indexes.dim() == 0:
            pred_indexes = pred_indexes.unsqueeze(0)
        if len(pred_indexes) <= max_step:
            max_step = len(pred_indexes)
        _,indexes = torch.topk(tensor_filtrado,max_step,largest=False)
        pred_indexes = pred_indexes[indexes]
        if pred_indexes.dim() == 0:
            pred_indexes = pred_indexes.unsqueeze(0)
        #Es importante retornar los indices de mayor a menor por que asi se pueden ir eliminando
        #de forma coherente.
        pred_indexes,_ = torch.sort(pred_indexes,descending=True)
        return pred_indexes

    def reduce_bulk(self,indexes: list[int]):
        for i, index in enumerate(indexes):
            self.instance = self.reduce(int(index))

    def reduce(self,index)->Instance:
        real_index = self.instance_mask[index]
        self.global_mask[real_index] = 0
        new_costs = self.pop_array(self.instance.costs,index)
        new_profits = self.pop_array(self.instance.profits,index)
        self.instance_mask = self.pop_array(self.instance_mask,index)
        return Instance(self.instance.n_items-1,self.instance.gamma,self.instance.budget,new_profits,new_costs,{},{})
        
        
    def step(self,threshold: float = 0.1,max_step: int = 300):
        start = time.time()
        self.treshold = threshold
        self.step_size = max_step
        partial_cont_sol = self.cached_cont_sol[self.instance_mask.tolist()]
        pred: torch.Tensor = self.heu.evaluate(self.instance,partial_cont_sol)
        self.actual_pred = pred
        indices = self.fix_from_pred(pred,max_step,threshold)
        self.reduce_bulk(indexes=indices)
        self.time += time.time() - start
        self.n_steps += 1
    
    def solve(self):
        start = time.time()
        indices = self.instance_mask
        set_indices = set(indices)
        full_indexes_set = set(range(self.o_instance.n_items))
        full_indexes = torch.tensor(list(full_indexes_set-set_indices))
        to_fix = torch.ones(self.o_instance.n_items)*-1
        to_fix[full_indexes] = 0
        solution = gurobi(self.o_instance,solver_config=SolverConfig(VAR_TYPE.BINARY,True,to_fix,None,None,False))
        return Solution(solution.o,solution.sol,time.time() - start + self.time)
        
        
