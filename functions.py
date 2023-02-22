import numpy as np
from Versicolor import Versicolor
from Virginica import Virginica
from Setosa import Setosa




def f(*args):
    with open("iris.csv", "r") as f:
        args[0][-1]*next(f)
        
        match args[0][-2]:
            case 1:
                a = np.array([Virginica(i.split(",")) for i in f.readlines() if i.split(",")[-1] == "virginica\n" or i.split(",")[-1] == "virginica"])
                return [a, 2, len(a)+1]

            case 2:
                a = np.array([Setosa(i.split(",")) for i in f.readlines() if i.split(",")[-1] == "setosa\n" or i.split(",")[-1] == "setosa"])
                return [a, args[0][0], 3, len(a)+len(args[0][0])+1]

            case 3:
                a = np.array([Versicolor(i.split(",")) for i in f.readlines() if i.split(",")[-1] == "versicolor\n" or i.split(",")[-1] == "versicolor"])
                return [a, args[0][0], args[0][1]]


def remove(total, sd, matrix=f(f(f([1, 1])))):
    """
    Função que remove de forma aleatória 50 objetos do todo e os guarda para o teste
    Entrada: matrix (=Composição de funções que gera uma matrix com todos os vetores do csv)
    saída: matrix (=matrix com os objetos restantes aleatórios); objts (=objetos removidos em ordem de remoção)
    """
    
    
    matrix_1d = list(np.array(matrix).reshape(sum([len(matrix[i]) for i in range(len(matrix))])))   
    objects = [set([np.random.choice(matrix[i], p=sd) for j in range(int(len(matrix[i])*total))]) for i in range(len(matrix))] # Retira 33% do total escolhido de cada espécie
    final_matrix = [[j for j in matrix[i] if j not in objects[i]] for i in range(len(matrix))]
    inde = [[matrix_1d.index(j) for j in objects[i]] for i in range(len(objects))]
    
    return [final_matrix, [inde[0]+inde[1]+inde[2]][0], matrix_1d, 1]