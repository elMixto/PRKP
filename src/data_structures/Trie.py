from itertools import combinations


class Nodo():
    def __init__(self) -> None:
        self.children: list[Nodo] = []
        self.value = 0


class KeyStorage:
    def __init__(self) -> None:
        pass

def calcular_hash_conjunto(conjunto):
    lista_ordenada = sorted(list(conjunto))
    hash_lista = hash(tuple(lista_ordenada))
    return hash_lista