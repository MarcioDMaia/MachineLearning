from numpy import *
from Vec import trans


class Flowers:

    def __init__(self, *args):
        self.args_num     =  list(map(float, args[0][:-1]))
        self.species      = args[0][-1]
        self.remove()
        self.norm         = linalg.norm(vectorize(trans)(self.args_num))
        self.axs_names    = ["Comprimento Sepala", "Largura sepala", "Comprimento petala", "Largura petala"]
        self.color_2d     =  self.color(self.species)

        
        
      
    def remove(self):
        if (self.species == "virginica\n") | (self.species == "setosa\n") | (self.species == "versicolor\n"):
            self.species = self.species[:-1]
            

    def color(self, specie):
        match specie:
            
            case "virginica":
                return "red,"
            case "setosa":
                return "blue,"
            case "versicolor":
                return "black,"
            

    def __it__(self, other):
        if self.norm > other.norm:
            return False
        else:
            return True

    def __gt__(self, other):
        if self.norm > other.norm:
            return True

        else: 
            return False
    
    def fo(self):
        return self.norm


    def lista(self, i):
        return self.args_num[i]