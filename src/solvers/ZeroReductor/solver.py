import torch
from torch import optim
import time
from pathlib import Path
import numpy as np
from src.data_structures.features import *
from src.data_structures import Instance, Solution
from src.solvers.ZeroReductor.DLHEU2 import DHEU
from src.Gurobi import SolverConfig,gurobi

class ZeroReductor2:
    def __init__(self,instance: Instance) -> None:
        self.o_instance = instance
        self.global_mask = [-1 for i in range(self.o_instance.n_items)]
        self.instance_mask = [i for i in range(self.o_instance.n_items)]
        self.instance: Instance  = self.o_instance
        self.pop_array = lambda array,index: np.concatenate([array[:index],array[index+1:]])
        self.features = [
            ProfitOverBudget(),LowerCostOverBudget(),
            UpperCostOverBudget(),IsInContSol(),
            #CountPSynergiesOverNItems(),CountPSynergiesOverBudget(),
            GammaOverNItems(),
            #SumOfSynergiesByItemOverMaxSinergyProfit(),
            Noise()]
        self.heu = DHEU(self.features)
        model = Path(__file__).resolve().parent / "models/DHEUV2.model"
        self.heu.load(model)
        self.heu.net.eval()
        #Prediccion inicial
        self.actual_pred: torch.Tensor = self.heu.evaluate(self.instance)
        self.step_size = 1
        self.treshold = 0.3
        self.time = 0
        self.n_steps = 0
    
    
    def get_reward(self):
        #A estas cosas eventualmente hay que pasarlas por otra funcion para mas o menos balancear
        #el peso de cada una, pero sin importar eso, el solver deberia terminar optimizando para ambas

        return torch.tensor([
            #1 - actual_sol/optimal_sol, se maximiza cuando actual_sol es igual al optmial_sol
            1 - (gurobi(self.instance,SolverConfig.optimal()).o  / self.o_instance.evaluate(self.o_instance.get_feature(IsInOptSol()))),
            #Idealmente deberia comparar con el tiempo que se demora el solver, pero 0.0, eso significa resolver todo denuevo, para ver cuando se demora
            1/self.time
            ])
    
    def get_actions(self):
        #New step_size, New threshold
        return torch.tensor([self.step_size//self.instance.n_items,self.treshold])


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
        pred_indexes,_ = torch.sort(pred_indexes,descending=True)
        return pred_indexes

    def reduce_bulk(self,indexes: list[int]):
        for i, index in enumerate(indexes):
            self.instance = self.reduce(int(index),0)

    def reduce(self,index,value)->Instance:
        real_index = int(self.instance_mask[index])
        self.global_mask[real_index] = value
        new_costs = self.pop_array(self.instance.costs,index)
        new_profits = self.pop_array(self.instance.profits,index)
        self.instance_mask = self.pop_array(self.instance_mask,index)
        new_gains = dict()
        for tupla,gain in self.instance.polynomial_gains.items():
            if index in tupla:
                continue
            else:
                new_tupla = []
                for element in tupla:
                    if element > index:
                        new_tupla.append(element-1)
                    else:
                        new_tupla.append(element)
                new_tupla.sort()
                new_tupla = str(tuple(new_tupla))
                new_gains[new_tupla] = gain
        output = Instance(self.instance.n_items-1,self.instance.gamma,self.instance.budget,new_profits,new_costs,new_gains,{})
        return output
        
    def step(self,threshold: float = 0.1,max_step: int = 300):
        start = time.time()
        self.treshold = threshold
        self.step_size = max_step
        pred: torch.Tensor = self.heu.evaluate(self.instance)
        self.actual_pred = pred
        indices = self.fix_from_pred(pred,max_step,threshold)
        self.reduce_bulk(indexes=indices)
        self.time += time.time() - start
        self.n_steps += 1
    
    def solve(self):
        from copy import deepcopy
        start = time.time()
        solution = gurobi(self.instance,solver_config=SolverConfig.optimal())
        output_global_mask = deepcopy(self.global_mask)
        for index, value in enumerate(solution.sol):
            real_element = int(self.instance_mask[index])
            output_global_mask[real_element] = value
        return Solution(self.o_instance.evaluate(output_global_mask),output_global_mask,time.time() - start + self.time)
        
        
