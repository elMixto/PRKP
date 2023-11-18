from pathlib import Path
import sys
import time
from src.data_structures import Instance
import numpy as np

from src.data_structures.features import CountPSynergiesOverBudget, CountPSynergiesOverNItems, GammaOverNItems, IsInContSol, ItemBatchFeature, LowerCostOverBudget, ProfitOverBudget, SumOfSynergiesByItemOverMaxSinergyProfit, UpperCostOverBudget
from src.solvers.DLHEU2 import DHEU
from src.solvers.collection import Solution, SolverCollection
import torch

from src.solvers.gurobi.gurobi_solver import SolverConfig
import torch.nn.utils.prune as prune




class ZeroReductor:
    def __init__(self,instance: Instance) -> None:
        self.o_instance = instance
        self.global_mask = [-1 for i in range(self.o_instance.n_items)]
        self.instance_mask = [i for i in range(self.o_instance.n_items)]
        self.pop_array = lambda array,index: np.concatenate([array[:index],array[index+1:]])
        self.instance = self.o_instance
    

    def fix_from_pred(self,pred: torch.Tensor,max_step: int):
        index_to_fix = torch.argmin(pred)
        mascara = torch.lt(pred, 0.3)
        pred_indexes = torch.nonzero(mascara).squeeze()
        
        tensor_filtrado = pred[pred_indexes]

        if pred_indexes.dim() == 0:
            pred_indexes = pred_indexes.unsqueeze(0)

        if len(pred_indexes) <= max_step:
            max_step = len(pred_indexes)

        values,indexes = torch.topk(tensor_filtrado,max_step,largest=False)
        pred_indexes = pred_indexes[indexes]
        if pred_indexes.dim() == 0:
            pred_indexes = pred_indexes.unsqueeze(0)
        pred_indexes,_ = torch.sort(pred_indexes,descending=True)
        return pred_indexes

    def reduce_bulk(self,indexes: list[int]):
        for i, index in enumerate(indexes):
            self.instance = self.reduce(int(index),0)


    def reduce(self,index,value)->Instance:
        if value == 0:
            #El elemento original que estoy fijando
            a = len(self.instance_mask)
            real_index = int(self.instance_mask[index])

            #Marco el valor que estoy fijando en la mascara global
            self.global_mask[real_index] = value

            #Le quito el elemento a los costos y profits
            new_costs = self.pop_array(self.instance.costs,index)
            new_profits = self.pop_array(self.instance.profits,index)
            
            #Le quito el elemento a la mascara
            self.instance_mask = self.pop_array(self.instance_mask,index)
            
            #Indica que elementos son los que estan siendo representados en el problema especifico
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
        if value == 1:
            pass#TODO
        else:
            pass
        
    def solve(self)->Solution:
        features: list[ItemBatchFeature] = [
            ProfitOverBudget(),
            LowerCostOverBudget(),
            UpperCostOverBudget(),
            IsInContSol(),
            CountPSynergiesOverNItems(),
            CountPSynergiesOverBudget(),
            GammaOverNItems(),
            SumOfSynergiesByItemOverMaxSinergyProfit()

            ]
        heu = DHEU(features)
        heu.load(Path("/home/mixto/repositories/PRKP/models/DHEUV2.model"))
        heu.net.eval()
        pred: torch.Tensor = heu.evaluate(self.instance)
        threshold = 0.1
        #Obtengo una prediccion inicial desde el modelo entrenado
        while torch.min(pred) < threshold:
            print(self.instance)
            start = time.time()
            pred: torch.Tensor = heu.evaluate(self.instance)
            end = time.time()-start
            #fixes = torch.tensor(fix_variables(self.instance.n_items,pred,percentage))
            #index_to_fix: torch.Tensor = torch.where(fixes == 0)[0]
            indices = self.fix_from_pred(pred,max_step=int(self.instance.n_items//np.log2(self.instance.n_items)))
            self.reduce_bulk(indexes=indices)
            sys.stdout.write(f"\r{self.instance.n_items} Actual pred:{torch.min(pred)} eval_time: {end}")
            sys.stdout.flush()
        
        #solution debe resolverse con un solver cualquiera
        solution = SolverCollection.gurobi(self.instance,solver_config=SolverConfig.optimal())
        print(solution.sol)
        for index, value in enumerate(solution.sol):
            real_element = int(self.instance_mask[index])
            self.global_mask[real_element] = value
        self.o_instance.evaluate(self.global_mask)
        return self.global_mask
