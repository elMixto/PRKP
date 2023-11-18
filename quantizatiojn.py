import torch
import torch.nn as nn
from matplotlib import pyplot as plt


class VectorQuantization(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(VectorQuantization, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1, 1)  # Inicialización de los embeddings

    def forward(self, x):
        # Redimensionar la entrada para que sea 2D (batch_size, seq_len)
        x = x.view(-1, self.embedding_dim)
        
        # Encontrar el índice más cercano en el diccionario de embeddings
        dist = torch.cdist(x.unsqueeze(0), self.embedding.weight.unsqueeze(0))
        indices = torch.argmin(dist, dim=-1)
        quantized = self.embedding(indices)
        
        # Reshape de vuelta a la forma original
        quantized = quantized.view(x.size())

        return quantized, indices

# Definir la clase VectorQuantization (como se proporcionó anteriormente)

# Generar datos unidimensionales de ejemplo
size = 100

torch.manual_seed(42)
data = torch.randn(size)  # Generamos 100 puntos de datos unidimensionales
x = torch.linspace(1,100,size)

# Crear una instancia de VectorQuantization con un número arbitrario de embeddings
num_embeddings = 10  # Número de embeddings
embedding_dim = 1  # Dimensión de cada embedding (1 para datos unidimensionales)
vq = VectorQuantization(num_embeddings, embedding_dim)

# Cuantizar los datos
quantized_data, indices = vq(data.unsqueeze(1))  # Agregar una dimensión para que sea 2D
import torch.nn.functional as F
loss = F.mse_loss(quantized_data, data.unsqueeze(1))
print(loss)


# Graficar los datos originales y cuantizados
plt.figure(figsize=(10, 6))
plt.scatter(x, data.numpy(), label='Datos Originales', alpha=0.5)
plt.scatter(x,quantized_data.squeeze().detach().numpy(), label='Datos Cuantizados', alpha=0.5)
plt.legend()
plt.xlabel('Valor')
plt.title('Comparación entre Datos Originales y Cuantizados')
plt.show()
