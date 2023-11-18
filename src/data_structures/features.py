from abc import ABC,abstractmethod
import numpy as np
from numpy.typing import ArrayLike
from torch import Tensor
from functools import lru_cache
import torch
#Si agrego las Features a data structures, voy a terminar creando dependencia circular

class ItemSingleFeature(ABC):
    """Item feature es una feature que se puede obtener para aun item individual de una instancia"""
    @property
    @abstractmethod
    def name(self)-> str:
        pass

    @staticmethod
    @abstractmethod
    def evaluate(instance: "Instance", item: int) -> float:
        """Esta funcion debe tomar una instancia y un item, y retornar un flotante"""
        pass


class ItemBatchFeature(ABC):
    """Item Batch Feature, es una feature que se puede obtener para todos los items de una instancia,
        en una sola operacion, como por ejemplo resolver la versión relajada del problema."""
    @property
    @abstractmethod
    def name(self)-> str:
        pass

    @abstractmethod
    def batch_evaluate(self,instance: "Instance") -> ArrayLike:
        """Esta funcion debe tomar una instancia y generar las features para toda la instancia"""
        pass        


### Features

class NItems(ItemBatchFeature):
    def __init__(self) -> None:
        super().__init__()

    name = "NItems"

    def batch_evaluate(self,instance: "Instance") -> ArrayLike:
        return torch.tensor([instance.n_items/10000 for i in range(instance.n_items)])


class Budget(ItemBatchFeature):
    name = "Budget"
    def batch_evaluate(self,instance: "Instance") -> float:
        """Normalized Budget?"""
        return torch.tensor([ instance.budget/instance.n_items  for i in range(instance.n_items)])


class Noise(ItemBatchFeature):
    """Para medir si otras features son tan utiles como el ruido"""
    name = "Noise"

    def batch_evaluate(self,instance: "Instance") -> ArrayLike:
        """Average of """
        salida = np.random.random(instance.n_items)
        return torch.tensor(salida)
    
    
class IsInContSol(ItemBatchFeature):
    """
    Resuelve la relajacion continua del problema de optimizacion y retorna la solucion para cada uno de los items, si se encuentra o no en la solucion
    (Usa el solver cacheado para poder reconstruir la trainingdata de forma rapida)
    """
    @property
    def name(self):
        return "IsInContSol"

    def batch_evaluate(self,instance: "Instance") -> ArrayLike:
        from src.Gurobi import SolverConfig,gurobi
        solution = gurobi(instance,SolverConfig.continous()).sol
        return torch.tensor(solution)
    
class IsInOptSol(ItemBatchFeature):
    """Resuelve la relajacion continua del problema de optimizacion y retorna la solucion para cada uno de los items, si se encuentra o no en la solucion"""
    @property
    def name(self):
        return "IsInOptSol"
    
    def batch_evaluate(self,instance: "Instance") -> ArrayLike:
        from src.Gurobi import SolverConfig,gurobi
        return torch.tensor(gurobi(instance,SolverConfig.optimal()).sol)


class ProfitOverBudget(ItemSingleFeature,ItemBatchFeature):
    name = "ProfitOverBudget"

    def batch_evaluate(self,instance: "Instance") -> ArrayLike:
        return torch.tensor(instance.profits) / instance.budget
    
    @staticmethod
    def evaluate(instance: "Instance", item: int) -> float:
        return instance.profits[item]/instance.budget
    
class LowerCostOverBudget(ItemSingleFeature,ItemBatchFeature):
    name = "LowerCostOverBudget"
    
    def batch_evaluate(self,instance: "Instance") -> ArrayLike:
        return torch.tensor(instance.costs[:,0]) / instance.budget
    
    @staticmethod
    def evaluate(instance: "Instance", item: int) -> float:
        return instance.costs[item,0]/instance.budget
    

class GammaOverNItems(ItemBatchFeature):
    name = "GammaOverNItems"
    
    def batch_evaluate(self,instance: "Instance") -> ArrayLike:
        return torch.ones(instance.n_items)* instance.gamma / instance.n_items
    
    
class UpperCostOverBudget(ItemSingleFeature,ItemBatchFeature):
    name = "UpperCostOverBudget"
    
    def batch_evaluate(self,instance: "Instance") -> ArrayLike:
        return torch.tensor(instance.costs[:,1]) / instance.budget
    
    @staticmethod
    def evaluate(instance: "Instance", item: int) -> float:
        return instance.costs[item,1]/instance.budget


class CountPSynergies(ItemBatchFeature):
    """Cuenta las sinergias postivas asociadas a un item especifico, y guarda en memoria los resultados para cada
    instancia que se pase."""

    name = "CountPSynergiesOverNItems"
    @lru_cache(maxsize=None)
    @staticmethod
    def syns(instance: "Instance"):
        syns = np.zeros(instance.n_items)
        for pol_gain,value in instance.polynomial_gains.items():
            for item in pol_gain:
                syns[item] += value
        syns = np.array(syns)
        return torch.tensor(syns)

    
    def batch_evaluate(self,instance: "Instance") -> ArrayLike:
        return CountPSynergies.syns(instance)
    
    @staticmethod
    def evaluate(instance: "Instance",item: int) -> ArrayLike:
        return CountPSynergies.syns(instance)[item,1]/instance.n_items
    


class CountPSynergiesOverBudget(ItemBatchFeature):
    """Cuenta las sinergias postivas asociadas a un item especifico, y guarda en memoria los resultados para cada
    instancia que se pase."""

    name = "CountPSynergiesOverBudget"
    def batch_evaluate(self,instance: "Instance") -> ArrayLike:
        return CountPSynergies().batch_evaluate(instance)/instance.instance.budget
    
class CountPSynergiesOverNItems(ItemBatchFeature):
    """Cuenta las sinergias postivas asociadas a un item especifico, y guarda en memoria los resultados para cada
    instancia que se pase."""

    name = "CountPSynergiesOverBudget"
    def batch_evaluate(self,instance: "Instance") -> ArrayLike:
        return CountPSynergies().batch_evaluate(instance)/instance.n_items


class SynergieReduction(ItemBatchFeature):
    pass


class SumOfSynergiesByItemOverMaxSinergyProfit(ItemBatchFeature):
    """Cuenta las sinergias postivas asociadas a un item especifico, y guarda en memoria los resultados para cada
    instancia que se pase."""

    name = "SumOfSynergiesByItemOverMaxSinergyProfit"
    
    def batch_evaluate(self,instance: "Instance") -> ArrayLike:
        total = torch.sum(torch.tensor([value for value in instance.polynomial_gains.values()]))
        syns = np.zeros(instance.n_items)
        for pol_gain,value in instance.polynomial_gains.items():
            for item in pol_gain:
                syns[item] += value
        syns = np.array(syns)
        return torch.tensor(syns)/total
    

class IsInFirstFactibleSol(ItemBatchFeature):
    """
    Resuelve  del problema de optimizacion y retorna la solucion para cada uno de los items, si se encuentra o no en la solucion
    (Usa el solver cacheado para poder reconstruir la trainingdata de forma rapida)
    """
    @property
    def name(self):
        return "IsInFirstFactibleSol"

    def batch_evaluate(self,instance: "Instance") -> ArrayLike:
        from src.Gurobi import SolverConfig,gurobi,VAR_TYPE
        solver_config = SolverConfig(var_type=VAR_TYPE.BINARY,heuristic=False,indexes=[],first_sol=True)
        solution = gurobi(instance,solver_config=solver_config).sol
        return torch.tensor(solution)


class NOverON(ItemBatchFeature):
    name = "NOverON"
    def __init__(self,o_instance: "Instance"):
        self.o_instance = o_instance
    
    def batch_evaluate(self,instance: "Instance") -> ArrayLike:
        return torch.ones(instance.n_items) * instance.n_items/self.o_instance.n_items

class GammaOverN(ItemBatchFeature):
    name = "GammaOverN"
    
    def batch_evaluate(self,instance: "Instance"):
        return torch.ones(instance.n_items) * instance.gamma/instance.n_items
    

class GammaOverON(ItemBatchFeature):
    name = "GammaOverON"
    
    def batch_evaluate(self,instance: "Instance"):
        return torch.ones(instance.n_items) * instance.gamma/instance.n_items    

    def __init__(self,o_instance: "Instance"):
        self.o_instance = o_instance
