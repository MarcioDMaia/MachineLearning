import numpy as np
from functions import remove
import warnings
from datetime import datetime
from Vec import ver2, plot
from Flowers import Flowers



    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

class Logic:
    def __init__(self, var_norm, now, total=0.33, seed=13, args=None):
        args = remove(sd=seed, total=total)
        self.norm_var        = var_norm # Variável necessária para diminuir o espaço amostral
        self.now             = now # Variável necessária para logging
        self.start           = datetime.now() # Variável necessária para o calculo do tempo total de cada interação
        self.matrix          = args[0] # Matriz separada por espécies 
        self.index           = args[1] # Matriz de indices que contem os objetos retirados
        self.original_matrix = args[2] # Matriz que contem os objetos retirados e não retirados
        self.rearranges() # Organiza a variável self.matrix em ordem crescente perante a norma
        self.falso_posi      = 0 # Variável necessária para retornar ao programa mandante dessa classe
        self.acuracy         = 0.0 # Variável necessária para retornar ao programa mandante dessa classe
        self.total_analisado = 0 # Variável necessária para retornar ao programa mandante dessa classe
        self.sepal_length    = self.rearranges(0) # Variável que armazena as informações da sepal_length no formato [min[ver, set, vir]],max[...],std[...]]
        self.sepal_width     = self.rearranges(1) # Variável que armazena as informações da sepal_width no formato [min[ver, set, vir]],max[...],std[...]]
        self.petal_length    = self.rearranges(2) # Variável que armazena as informações da petal_length no formato [min[ver, set, vir]],max[...],std[...]]
        self.petal_width     = self.rearranges(3) # Variável que armazena as informações da petal_width no formato [min[ver, set, vir]],max[...],std[...]]
        self.dict            = {
                                "Versicolor":{"versicolor":0, "setosa":0, "virginica":0},
                                "Setosa": {"versicolor":0, "setosa":0, "virginica":0},
                                "Virginica": {"versicolor":0, "setosa":0, "virginica":0},
                                "Total": 0
                                } # Matrix de confusão
        self.ver = self.__update(0)
        self.set = self.__update(1)
        self.set = self.__update(2)
        self.prev()
        self.display()      
        



    def file(self, *args):
        """Método para fechar e/ou abrir o arquivo logging novamente
            Entrada: args[text]
            Saída: Void
        """ 
        log = open("logging.txt", "a") # Abre o arquivo desejado
        log.write(f"{args[0]}\n") # Escreve o conteúdo desejado no arquivo
        log.close() # O fecha

    def display(self): # Método principal, tem como objetivo, girar a engranagem do funcionamento do programa
        while True:
            print("[1] – Mostrar  informações\n[2] – Mostrar gráficos\n[3] – Classificar amostras\n[4] - Adicionar ponto aleatório\n[5] – Sair")
            try:   
                match int(input()):
                    case 1: # update ja funcionando -- Printa todas as informações que foi pedido sobre as espécies 
                        self.ver = self.__update(0)
                        self.set = self.__update(1)
                        self.vir = self.__update(2)
                        self.stri(13) # 13 == melhor numero

                    case 2:
                        inf = "Digite de 0 a 3 para escolher as informações do gráfico:\n"
                        match int(input("Digite quantas dimensões deseja (de 2 a 4)\n")):
                            case 2:
                                plot((int(input(inf)), int(input(inf))), self.matrix)

                            case 3:
                                plot((int(input(inf)),int(input(inf)), int(input(inf))), self.matrix)

                            case 4:
                                pass
                        

                    case 3: # Previsão de novos objetos
                        self.start = datetime.now()
                        inf = ["Comprimento da Sepala:\n ", "Largura da Setala:\n", "Comprimento da Petala:\n", "Largura da Petala:\n"]
                        object = []
                        counter = 0
                        while True:
                            object.append(float(input(inf[counter])))
                            if (counter == 3):
                                object.append("generic")
                                break
                            counter += 1

                        a = self.prevn(object)
                        print(f"A espécie analisada tem {self.acuracy*100}% de chance de ser {a}\n{self.dict}")


                    case 4:
                        qt = int(input("Quantas espécies aleátórias você quer?\n"))
                        while True:
                            obj = np.random.uniform(1, 8, (1,4)).tolist()
                            obj = [abs(i) for i in obj[0]]
                            obj.append("generic")
                            a = self.prevn(obj)
                            print(f"O {qt} tem {self.acuracy*100}% de ser {a}")
                            qt -= 1
                            if qt == 0:
                                break

                    
                    case 5:
                        break

                    case _:
                        print("Digite um número de 1 a 5")

                
                        
            except ValueError:
                print("Digite apenas números")
                continue



    def prev(self):
        counter = 0

        while True:
            self.rearranges() # Reorganiza a matrix atualizada conforme a ordem crescente das normas
            teste = self.original_matrix[self.index[counter]] # Atraves da lista de indices aleatorios (self.index), retira o objeto que se deseja prever
            counter+=1
            m = ver2([[sum([(teste.lista(k)-j.lista(k))**2 for k in range(len(j.args_num)) if teste.fo()-self.norm_var <= j.fo() <= teste.fo()+self.norm_var])**(1/2) for j in i]for i in self.matrix]) # Calcula a média das espécies que se encontram a  norma_novo_vetor +- 0.4
        
            if ((m[0] < m[1]) & (m[0] < m[2])): # Através das menores médias entre os grupos inteiros, pressupoem a qual grupo esse novo objeto pertence
                if (teste.species != "versicolor"): # Casos onde virginica não tinha quebra de linha e estava dando falso, falso positivo
                    self.falso_posi +=1 # Acrescido um sempre que a previsão é incorreta
                    self.dict["Versicolor"][f"{teste.species}"] += 1 # Adiciona a matriz de confusão qual foi o erro cometido
                    
                else:                    
                    self.matrix[0].append(teste) # Adiciona ao banco de dados o novo objeto previsto com sucesso
                    self.dict["Versicolor"][f"{teste.species}"] += 1
                    

            if ((m[1] < m[0]) & (m[1] < m[2])): # Através das menores médias entre os grupos inteiros, pressupoem a qual grupo esse novo objeto pertence
                if (teste.species != "setosa"): 
                    self.falso_posi += 1 # Acrescido um sempre que a previsão é incorreta
                    self.dict["Setosa"][f"{teste.species}"] += 1 # Adiciona a matriz de confusão qual foi o erro cometido

                else:                    
                    self.matrix[1].append(teste) # Adiciona ao banco de dados o novo objeto previsto com sucesso
                    self.dict["Setosa"][f"{teste.species}"] += 1
                
            if ((m[2] < m[1]) & (m[2] < m[0])): # Através das menores médias entre os grupos inteiros, pressupoem a qual grupo esse novo objeto pertence
                if (teste.species != "virginica"): # Casos onde virginica não tinha quebra de linha e estava dando falso, falso positivo
                    self.falso_posi +=1 # Acrescido um sempre que a previsão é incorreta
                    self.dict["Virginica"][f"{teste.species}"] += 1 # Adiciona a matriz de confusão qual foi o erro cometido
                    
                else:
                    self.matrix[2].append(teste) # Adiciona ao banco de dados o novo objeto previsto com sucesso
                    self.dict["Virginica"][f"{teste.species}"] += 1

                    
            self.dict["Total"] += 1 # Adiciona um ao total a cada interação >> Alterar depois para self.dict["Total"] = len(self.index)

            if (counter == len(self.index)):
                    self.acuracy =1-(self.falso_posi/len(self.index))
                    self.file(f"Teste numero: {self.now}   Falso positivo: {self.falso_posi}   Precisao: {self.acuracy}   tamanho indices:   {len(self.index)}   Total : {counter}   Tempo de execucao unitario: {datetime.now()-self.start}\n\n") # Salva informações necessárias no arquivo logging


            if (counter == len(self.index)): # Para o loop caso tenha previsto todos os casos exluídos aleatoriamente
                self.acuracy =1-(self.falso_posi/len(self.index))  # Calcula a precisão do caso em si
                self.total_analisado = len(self.index) # Salva quantos casos foram analisados para entregar ao programa de controla este arquivo
                break    

    def prevn(self, ob):
        prev = ""
        teste = Flowers(ob)
        m = [sum([sum([(teste.lista(k)-j.lista(k))**2 for k in range(len(j.args_num))])**(1/2) for j in i])for i in self.matrix] # Calcula a média das espécies que se encontram a  norma_novo_vetor +- 0.4
        m = [m[i]/len(self.matrix[i]) for i in range(len(m))]
        if ((m[0] < m[1]) & (m[0] < m[2])):
            teste.species = "versicolor"
            self.matrix[0].append(teste)
            prev = "Versicolor"

        if ((m[1] < m[0]) & (m[1] < m[2])):
            teste.species = "setosa"
            self.matrix[1].append(teste)
            prev = "Setosa"

        if ((m[2] < m[1]) & (m[2] < m[0])):
            teste.species = "virginica"
            self.matrix[2].append(teste)
            prev = "Virginica"
        return prev



    def __update(self, command):
        """ 
        Metodo privado: Separa as estatisticas de cada especie
        Entrada: command(in [0,3] c Z (0=versitosa, 1=setosa, 2=virginica))
        Saída: list(sepla_length(min, max, std), sepal_width(min,...,std), petal_length(...), ...(...))
        """
        self.inf = np.array([self.rearranges(0), self.rearranges(1), self.rearranges(2), self.rearranges(3)])
        return [[i[j][command] for j in range(len(self.inf)-1)] for i in self.inf]

    def stri(self, command=0):
        """Método que stri, retorna uma string contendo as informações de cada espécie
            Entrada: command (=0: min_versicolor(sepal_length, sepal_width, ..., petal_width), max_ver(sepal_length,...), std_ver(...);
                    =1: min_set(...), ..., std_...(...);
                    =2: min_vir(...),...;
                    =any: [0,1,2])

            Saída: Void
        """
    
        list_inf = [
                    ["Min Sepal Length: ", "Max Sepal Length: ", "Std Sepal Length: "],
                    ["Min Setal Width: ", "Max Sepal Width: ", "Std Sepal Width: "],
                    ["Min Petal Length: ", "Max Petal Length: ", "Std Petal Length: "],
                    ["Min Petal Width: ", "Max Petal Width: ", "Std Petal Width: "]
                ]

        str_inf = []

        match command:
            case 0:
                str_inf = [[f"{list_inf[i][j]}{self.ver[i][j]}\n" for j in range(len(self.ver[command]))] for i in range(len(self.ver))]
                print("Versicolor:\n\n")

            case 1:
                str_inf = [[f"{list_inf[i][j]}{self.set[i][j]}\n" for j in range(len(self.set[command]))] for i in range(len(self.set))]
                print("Setosa:\n\n")

            case 2:
                str_inf = [[f"{list_inf[i][j]}{self.vir[i][j]}\n" for j in range(len(self.vir[command]))] for i in range(len(self.vir))]
                print("Virginica: \n\n")

            case _:
                self.stri(0)
                self.stri(1)
                self.stri(2)
        
        [print("".join(i)) for i in str_inf]
        print("\n\n")


    def rearranges(self, command="norm"):
        """
        Método rearranges: Organiza a matriz original com base no comando ou retorna o max, min e o desvio padrão de todas as espécies
        Entrada: command(=norm, 0=sepal_length, 1=sepal_width, 2=petal_length, 3=petal_width)
        Saída: command(=norm: Void; in [0,3]: array(min(versicolor, setosa, virginica)), array(max(versicolor,...)), array(std(...)))
        """
        match command:
            
            case "norm": 
                self.matrix = np.array([sorted(self.matrix[i]) for i in range(len(self.matrix))])

            case _:
                a = np.array([np.array([i.args_num[command] for i in self.matrix[j]]) for j in range(len(self.matrix))])
                return np.array([[np.min(a[i]) for i in range(len(a))], [np.max(a[i]) for i in range(len(a))], [np.std(a[i]) for i in range(len(a))]])
