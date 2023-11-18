from abc import ABC, abstractmethod
from .Instance import Instance
from dataclasses import dataclass
from enum import Enum
from hashlib import sha1

@dataclass
class Result(Enum):
    run_time: float

@dataclass
class StopReason(Enum):
    pass

class Heuristic(ABC):
    
    @abstractmethod
    def prepare():
        """
            Esta funciona prepara la heuristica, ya sea para generar datos preprocesados,
            entranr modelos etc
        """
        pass

    @abstractmethod
    def run()->StopReason:
        """Esta funcion ejecuta la heuristica hasta que finaliza, """
        pass
