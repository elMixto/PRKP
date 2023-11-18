import torch
from src.solvers.collection import Solution
import numpy as np

class ReductableInstance:
    #The typehints are there for my sake, in reality all vars will be tensors
    def __init__(self,
                 items: list[int],
                 gamma: int,
                 budget: float,
                 profits: list[float],
                 upper_costs: list[float],
                 nominal_costs: list[float],
                 polinomial_gains) -> None:
            pass
        
    def evaluate(self,sol):
        """Faster eval function to train a model :D"""
        synSet = self.synSet
        sol = np.array(sol,dtype=np.uint32)
        investments = np.multiply(sol,self.item_vect)
        investments = np.array([ int(i) for i in investments if i > 0 ]) - 1
        investments = np.sort(investments)
        investments = investments.tolist()
        investments.sort(key = lambda x: self.costs_deltas[x], reverse = True)
        investments = np.array(investments)
        upperCosts = np.sum([self.upper_costs[x] for x in investments[:self.gamma]])
        nominalCosts = np.sum([self.nominal_costs[x] for x in investments[self.gamma:]])
        total_costs = upperCosts + nominalCosts
        if total_costs <= self.budget:
            of = np.sum([self.profits[x] for x in investments]) - total_costs
            investments=set(investments)
            for i,syn in enumerate(synSet): #Por cada llave en polsyns
                if syn.issubset(investments): #Checkeo si la llave es subconjunto de los investments
                    of += self.polynomial_gains[self.pol_keys[i]] #Y si es subconjunto, retorno la cosa
            return of
        return -1