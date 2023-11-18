from abc import ABC,abstractmethod
import torch
from torch import absolute
from kmeans_pytorch import kmeans



class IntQuantizer(ABC):
    
    @abstractmethod
    def quantized(self)->list[float]:
        pass
    
    @abstractmethod
    def indexes(self)->list[int]:
        pass
    
    @abstractmethod
    def quantize(self,value,iters)->list[int]:
        pass
    @abstractmethod
    def cluster_centers(self):
        pass

class KmeansQuantizer(IntQuantizer):
    def __init__(self,data: list[float],num_clusters: int) -> None:
        self.cluster_ids_x, self._cluster_centers = kmeans(X=data, num_clusters=num_clusters, distance='euclidean', device=torch.device('cpu'))
        self._quantized = torch.index_select(self._cluster_centers,0,self.cluster_ids_x)
        self._indexes = self.cluster_ids_x

    def cluster_centers(self):
        return self._cluster_centers

    def quantized(self) -> list[float]:
        return self._quantized
    
    def indexes(self) -> list[int]:
        return self._indexes
    
    def quantize(self,value: torch.Tensor,iters: int = 2) -> (float,int):
        diferencias = torch.abs(self.quantized().view(1, -1) - value)
        indices_cercanos = self.indexes()[diferencias.argmin(dim=1)]
        return self.cluster_centers[indices_cercanos], indices_cercanos



class EmbeddingQuantizer(IntQuantizer):
    def __init__(self,data,lr,embedings_num,quantizer_epochs,embedding_dim) -> None:
        from src.solvers.OG.model import VectorQuantization
        from torch import nn
        from torch import optim
        from random import shuffle
        
        sample_size = 100
        quantizer_epochs = 10
        self.benefit_quantizer = VectorQuantization(num_embeddings=embedings_num, embedding_dim=embedding_dim)
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.benefit_quantizer.parameters(), lr= lr)
        train_data = data.T.detach().numpy().tolist()[0]
        shuffle(train_data)
        train_data = train_data[:sample_size]
        train_data = torch.tensor(train_data).view(-1,1)
        
        # Ciclo de entrenamiento 
        # (Basicamente estoy entrenando los embeddings  por si solos para recordar las cosillas)
        for _ in range(quantizer_epochs):
            total_loss = 0
            for inp in train_data:
                self.optimizer.zero_grad()
                quantized, _ = self.benefit_quantizer(inp)
                loss = self.criterion(quantized[0], inp)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        print(total_loss)
        self.quantized, self.indices = self.benefit_quantizer(data)
    
    def cluster_centers(self):
        return self.benefit_quantizer.embedding.state_dict()['weight'].T[0]
    def quantized(self) -> list[float]:
        return self.quantized
    
    def indexes(self) -> list[int]:
        return self.indices
    
    def quantize(self, value, iters) -> list[int]:
        return self.benefit_quantizer.embedding(value)




    