from src.data_structures import Instance
import torch


class ZeroInstance:
    def __init__(self,instance: Instance):
        self.items = torch.arange(0,instance.n_items,1)
        self.gamma = instance.gamma
        self.budget = instance.budget
        self.polynomial_gains = instance.polynomial_gains
        self.costs = torch.tensor(instance.costs)
        self.l_costs = self.costs[:,0]
        self.u_costs = self.costs[:,1]
        self.profits = torch.tensor(instance.profits)

        






    