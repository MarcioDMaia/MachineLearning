# Arquivo completar para o método vectorize do numpy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

def trans(num): return float(num)

def norm(vec): return vec.np.norm



def ver2(x):
    lista = []
    for i in range(len(x)):
        sum = 0
        counter = 0
        for j in range(len(x[i])):
            if x[i][j] == 0.0:
                continue
            else:
                sum += x[i][j]
                counter += 1
        try:
            lista.append(sum/counter)
        except ZeroDivisionError:
            lista.append(10000)
    return lista    


def plot(*args):
    """ Função que plota os gráficos desejados
        Entradas:  (=args[0] - tupla com as colunas desejadas a plotar, =args[1] - matriz de objetos)
        Saída: Vazia

    """
    matrix = args[1]
    nupla = args[0] 
    save = np.arange(1000)
    format = ["*", "d", "p"]
    plt.style.use("ggplot")

    match len(nupla):
        case 2:
            x = [[j.args_num[nupla[0]] for j in i]for i in matrix]
            y = [[j.args_num[nupla[1]] for j in i]for i in matrix]            
            fig, axs = plt.subplots(1,1, figsize=(11,4))
            axs.set_title(f"{matrix[0][0].axs_names[nupla[0]]} x {matrix[0][0].axs_names[nupla[1]]}")
            axs.set_xlim(0, max([max(i) for i in x])+1) 
            axs.set_ylim(0, max([max(i) for i in y])+1) 

            for i in range(len(x)):
                axs.scatter(x[i], y[i], label=f"{matrix[i][0].species}")
                axs.legend(loc="best")
            axs.set_xlabel(f"{matrix[0][0].axs_names[nupla[0]]}")
            axs.set_ylabel(f"{matrix[0][0].axs_names[nupla[1]]}")
            fig.savefig(f"{np.random.choice(save)}")
            plt.show()



        case 3:
            x = [[j.args_num[nupla[0]] for j in i]for i in matrix]
            y = [[j.args_num[nupla[1]] for j in i]for i in matrix]            
            z = [[j.args_num[nupla[2]] for j in i]for i in matrix]
            fig = plt.figure(figsize=(11,5))
            axs = fig.add_subplot(projection="3d")
            for i in range(len(matrix)):
                axs.set_title(f"{matrix[0][0].axs_names[nupla[0]]} x {matrix[0][0].axs_names[nupla[1]]} x {matrix[0][0].axs_names[nupla[2]]}")
                axs.scatter(x[i], y[i], z[i], label=f"{matrix[i][0].species}", marker=format[i])
                axs.legend(loc="best")
                
            axs.set_xlabel(f"{matrix[0][0].axs_names[nupla[0]]}")
            axs.set_ylabel(f"{matrix[0][0].axs_names[nupla[1]]}")
            axs.set_zlabel(f"{matrix[0][0].axs_names[nupla[2]]}")
            axs.set_xlim(0, max([max(i) for i in x])+1) 
            axs.set_ylim(0, max([max(i) for i in y])+1) 
            axs.set_zlim(0, max([max(i) for i in z])+1) 
            fig.savefig(f"{np.random.choice(save)}")
            plt.show()

        case 4:
            x      = [[j.args_num[nupla[0]] for j in i]for i in matrix]
            y      = [[j.args_num[nupla[1]] for j in i]for i in matrix]            
            z      = [[j.args_num[nupla[2]] for j in i]for i in matrix]
            colors = [[j.args_num[nupla[3]] for j in i]for i in matrix]
            fig = plt.figure()
            axs = fig.add_subplot(projection="3d")
            for i in range(len(matrix)):
                axs.set_title(f"{matrix[0][0].axs_names[nupla[0]]} x {matrix[0][0].axs_names[nupla[1]]} x {matrix[0][0].axs_names[nupla[2]]} x {matrix[0][0].axs_names[nupla[3]]}")
                axs.scatter(x[i], y[i], z[i], c=colors[i], cmap="inferno" ,label=f"{matrix[i][0].species}")
                axs.legend(loc="best")
                
            axs.set_xlabel(f"{matrix[0][0].axs_names[nupla[0]]}")
            axs.set_ylabel(f"{matrix[0][0].axs_names[nupla[1]]}")
            axs.set_zlabel(f"{matrix[0][0].axs_names[nupla[2]]}")
            axs.set_xlim(0, max([max(i) for i in x])+1) 
            axs.set_ylim(0, max([max(i) for i in y])+1) 
            axs.set_zlim(0, max([max(i) for i in z])+1) 
            plt.colorbar()
            plt.show()            