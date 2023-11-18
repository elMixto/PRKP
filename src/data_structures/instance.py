import logging
import random
from pathlib import Path
from dataclasses import dataclass, field
import math
import json
from hashlib import sha1
import numpy as np
from dataclasses_json import dataclass_json
from functools import lru_cache
from time import time
from numpy.typing import ArrayLike,NDArray
import torch
from torch import Tensor
from copy import deepcopy

class Instance:
    from src.data_structures.features import ItemBatchFeature,ItemSingleFeature
    def __init__(self,
                 n_items: int,
                 gamma: int,
                 budget: float,
                 profits: Tensor, #Tensor unidimensional de floats
                 costs: Tensor ,#Tensor unidimensional de floats
                 polynomial_gains: dict[set[int],int],
                 features: dict[str,list[float]]): 
        #Features es el vector que guarda las features, con el nombre de la feature y los valores para cada elemento
        #Variables asociadas al problema
        self.features = features
        self.n_items = n_items
        self.budget = budget
        self.gamma = gamma
        self.profits = profits
        self.costs = costs
        self.polynomial_gains = dict()
        for key,value in polynomial_gains.items():
            tuple_key = self.key_to_tuple(key)
            self.polynomial_gains[tuple_key] = value
        #Algunas cosillas precalculadas
        self.item_vect = np.linspace(1,self.n_items,self.n_items) #[1,2,3,4,5,6,....,n-1,n]
        #Los costos nominales separados
        self.nominal_costs = np.array(self.costs[:,0])
        #Costos maximo
        self.upper_costs = np.array(self.costs[:,1])
        self.total_nominal_costs = np.sum(self.nominal_costs)
        #Diferencia de los costos
        self.costs_deltas = np.add(self.upper_costs,-1 * self.nominal_costs)
    
    
    def get_feature(self,feature: ItemBatchFeature):
        if feature.name in self.features.keys():
            return torch.tensor(self.features[feature.name])
        output = feature.batch_evaluate(self)
        self.features[feature.name] = output.numpy().tolist()
        return output
    
        
    @staticmethod
    def key_to_tuple(key)-> tuple[int]:
        """Esta cosa transforma un string de la forma '(1,2,3,4)' a una tupla ordenada de la misma forma """
        number_list = key.replace("(","").replace(")","").replace("'","").split(",")
        number_list = [int(i) for i in number_list]
        number_list.sort()
        return tuple(number_list)
    
    @staticmethod
    def tuple_to_key(tupla: tuple):
        items = map(str,tupla)
        values = ",".join(items)
        return f"({values})"


    def evaluate(self,sol):
        """Faster eval function to train a model :D"""
        synSet = self.polynomial_gains.keys()
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
                if set(syn).issubset(investments): #Checkeo si la llave es subconjunto de los investments
                    of += self.polynomial_gains[syn] #Y si es subconjunto, retorno la cosa
            return of
        return -1
    
    
    @classmethod
    def from_file(cls,json_file):
        """Construye una instancia desde una archivo"""
        with open(json_file,"r",encoding="utf8") as f:
            json_file = json.load(f)
        output = cls.from_dict(json_file)
        return output

    @classmethod
    def from_dict(cls,json_file: dict):
        """Carga la instancia desde un diccionario (serializado desde json)"""
        logging.info("Loading instance")
        n_items = json_file['n_items']
        gamma = json_file['gamma']
        budget = json_file['budget']
        profits = np.array(json_file['profits'])
        costs = np.array(json_file['costs'])
        polynomial_gains = json_file['polynomial_gains']
        features = json_file['features']
        instance = cls(n_items,gamma,budget,profits,costs,polynomial_gains,features)
        return instance

    def save(self, folder_path: str | Path) -> str:
        """
        Guarda la instancia en una ruta 
        (Para guardar en el directorio de trabajo usar \"/\")
        
        Se serializa todo y luego se escribe en el archivo
        """
        json_output = self.to_json_string()
        file_path = Path(folder_path) / f"In{self.n_items}g{self.gamma}#{self.hash()}.json"
        with open(file_path, "w", encoding="utf-8") as json_file:
            json_file.write(json_output)
        return file_path

    def to_json_string(self) -> str:
        """
            Esta funcion retorna un json de la instancia, en un formato,
            que pueda cargarse en el futuro, y además incluye la solucion optima.
        """
        output = {}
        output['n_items'] = self.n_items
        output['gamma'] = self.gamma
        output['budget'] = self.budget
        output['profits'] = self.profits.tolist()
        output['costs'] = self.costs.tolist()

        output['polynomial_gains'] = dict()
        for key,value in self.polynomial_gains.items():
            output['polynomial_gains'][self.tuple_to_key(key)] = value
        
        output['features'] = dict()

        for key,value in self.features.items():
            if isinstance(value,list):
                output['features'][key] = value
            else:
                output['features'][key] = value.numpy().tolist()
        return json.dumps(output)

    @classmethod
    def generate(cls,n_items: int, gamma: int, seed = None)-> 'Instance':
        """Gamma is generally int(random.uniform(0.2, 0.6) * el)"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        matrix_costs = np.zeros((n_items, 2), dtype=float)
        d = [0.3, 0.6, 0.9]
        for i in range(n_items):
            matrix_costs[i, 0] = random.uniform(1, 50)
            matrix_costs[i, 1] = (1 + random.choice(d)) * matrix_costs[i, 0]

        array_profits = np.zeros((n_items), dtype=float)
        for i in range(n_items):
            array_profits[i] = random.uniform(0.8 * np.max(matrix_costs[:, 0]), 100)

        m = [2, 3, 4]
        budget = np.sum(matrix_costs[:, 0]) / random.choice(m)
        items = list(range(n_items))

        polynomial_gains = {}
        n_it = 0
        for i in range(2, n_items):
            if n_items > 1000:
                for j in range(int(n_items / 2 ** ((i - 1)))):
                    n_it += 1
                    elem = str(tuple(np.random.choice(items, i, replace=False)))
                    polynomial_gains[elem] = random.uniform(1, 100 / i)
            elif n_items <= 1000 and n_items > 300:
                for j in range(int(n_items / 2 ** (math.sqrt(i - 1)))):
                    n_it += 1
                    elem = str(tuple(np.random.choice(items, i, replace=False)))
                    polynomial_gains[elem] = random.uniform(1, 100 / i)
            else:
                for j in range(int(n_items / (i - 1))):
                    n_it += 1
                    elem = str(tuple(np.random.choice(items, i, replace=False)))
                    polynomial_gains[elem] = random.uniform(1, 100 / i)
        matrix_costs = matrix_costs.reshape(n_items, 2)
        return Instance(n_items,gamma,budget,array_profits,matrix_costs,polynomial_gains,dict())
    

    @classmethod
    def generate_quadratic(cls,n_items: int, gamma: int, seed = None)-> 'Instance':
        """Gamma is generally int(random.uniform(0.2, 0.6) * el)"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        matrix_costs = np.zeros((n_items, 2), dtype=float)
        d = [0.3, 0.6, 0.9]
        for i in range(n_items):
            matrix_costs[i, 0] = random.uniform(1, 50)
            matrix_costs[i, 1] = (1 + random.choice(d)) * matrix_costs[i, 0]

        array_profits = np.zeros((n_items), dtype=float)
        for i in range(n_items):
            array_profits[i] = random.uniform(0.8 * np.max(matrix_costs[:, 0]), 100)

        m = [2, 3, 4]
        budget = np.sum(matrix_costs[:, 0]) / random.choice(m)
        polynomial_gains = {}
        for i in range(n_items):
            for j in range(n_items):
                if i == j:
                    continue
                tupla = [i,j]
                tupla.sort()
                tupla = tuple(tupla)
                if np.random.random() >= 0.9:
                    polynomial_gains[str(tupla)] = np.random.random()
        matrix_costs = matrix_costs.reshape(n_items, 2)
        return Instance(n_items,gamma,budget,array_profits,matrix_costs,polynomial_gains,dict())
    
    def _id(self):
        return str(sha1(self.to_json_string().encode()).hexdigest())

    def hash(self) -> int:
        """
            El hash se obtiene del string que representa todos los parametros, sin tomar en cuenta
            los valores de la solucion optima.
            Idealmente cachear solo las cosas que cuestan mas en computar que este hash
        """

        output = {}
        output['n_items'] = self.n_items
        output['gamma'] = self.gamma
        output['budget'] = self.budget
        output['profits'] = self.profits.tolist()
        output['costs'] = self.costs.tolist()
        
        output['polynomial_gains'] = dict()
        for tupla,value in self.polynomial_gains.items():
            output['polynomial_gains'][self.tuple_to_key(tupla)] = value

        return str(sha1(json.dumps(output).encode(),usedforsecurity=False).hexdigest())

    def __str__(self) -> str:
        return f"Instance({self.n_items},{self.gamma},#{hash(self)})"


@dataclass
class Solution:
    """Clase simple para almacenar soluciones"""
    o: float
    sol: ArrayLike
    time: float
    def __repr__(self) -> str:
        return f"Sol(of:{self.o},time:{self.time})"