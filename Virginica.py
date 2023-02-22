from Flowers import Flowers
from numpy import *


class Virginica(Flowers):
    def __init__(self,*args):   
        super().__init__(args[0]) # Neste caso, o tipo da flor (str) se encontra no ultimo indice dos argumentos
        