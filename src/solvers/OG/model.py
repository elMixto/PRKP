import torch
from torch import nn
import numpy as np
import sys
import matplotlib.pyplot as plt
from torch import nn
import torch.optim as optim
from src.data_structures import instance
from src.solvers.OG.integer_quantizer import KmeansQuantizer,IntQuantizer
from functools import partial
from random import shuffle



class VectorQuantization(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(VectorQuantization, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1, 1)

    def forward(self, x):
        x = x.view(-1, self.embedding_dim)
        dist = torch.cdist(x.unsqueeze(0), self.embedding.weight.unsqueeze(0))
        indices = torch.argmin(dist, dim=-1)
        quantized = self.embedding(indices)
        quantized = quantized.view(x.size())
        return quantized, indices


class Net(nn.Module):
    def __init__(self, entrada,salida):
        super(Net, self).__init__()

        hidden_size = 100

        self.many = nn.Sequential(
            nn.Linear(entrada, hidden_size),
            nn.Linear(hidden_size,salida),
            nn.Sigmoid())

    def forward(self, x):   
        x = self.many(x)
        return x

def tupla_a_vector_binario(tupla,size):
    output = np.zeros(size)
    for i in tupla:
        output[i] = 1
    return output



class CustomSolver:
    """El solver se construye para cada instancia individualmente,
       no es un solver que se construya y funcione para todas las intancias
    """
    def __init__(self,instance: instance) -> None:
        self.instance = instance
        self.size = len(instance.polynomial_gains)
        self.benefit_data = torch.tensor(list(instance.polynomial_gains.values())).view(-1,1)
        self.tupla_a_vector_binario = partial(tupla_a_vector_binario,size=self.instance.n_items)
        #Benefit_data debe tener la forma [[1.0],[0.5],[3.5]] por cada beneficio para ser pasados al modelo para entrenamiento
    
    def set_integer_quantizer(self,quantizer: IntQuantizer ):
        self.quantizer = quantizer
        
    def create_vectorized_data(self):
        return torch.tensor(np.array(list(map(self.tupla_a_vector_binario,self.instance.synSet)),dtype=np.float64))

    
    def benef_from_indices(self,indices: torch.Tensor)-> torch.Tensor:
        """Los indices deben tener la forma [[1,2,3,4]] y retorna """
        return self.benefit_quantizer.embedding(indices[0])
