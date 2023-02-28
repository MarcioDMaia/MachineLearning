# Inteligência artificial (KNN)

## Sumário
1. Apresetação
2. Introdução
3. Objetivo
4. Dilema
5. Confecção

    -5.1 Treinamento

        -5.1.1 Função Gamma

        -5.1.2 Função Theta

        -5.1.3 Função Omega

        -5.1.4 Matriz de confusão 

    -5.2 Novos vetores

    -5.3 Tratamento do banco de dados

---

## 1. [Apresentação](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
>O modelo de KNN é um método de apendizado supervisionado não paramétrico, desenvolvido pela primeira vez por  [Evelyn Fix (ref. 1.1)](https://en.wikipedia.org/wiki/Evelyn_Fix) e [Joseph Hodges (ref. 1.2)](https://en.wikipedia.org/wiki/Joseph_Lawson_Hodges_Jr.) em 1951 e posteriormente expandido por [Thomas M. Cover (ref. 1.3)](https://en.wikipedia.org/wiki/Thomas_M._Cover). É usado para classificação e regressão. Em ambos os casos, a entrada consiste nos k exemplos de treinamento mais próximos em um conjunto de dados . A saída depende se k -NN é usado para classificação ou regressão.


---

## 2. Introdução
    
---
## 3. Objetivo
    Criar uma inteligência artificial baseada no modelo KNN (K-Nearest Neighbors), para a identificação de espécies distitas de iris.

        - Iris Versicolor
        - Iris Virginica
        - Iris Setosa


    Para a confexão do projeto, foi proposto o uso da distância euclidiana que segue a seguinte forma:


Sejam dois objetos: $X_1 = (x_{11}, x_{12},\dots, x_{1n})$ e $X_2 = (x_{21}, x_{22},\dots, x_{2n})$ a distância entre tais pontos é dada pela fórmula da distância euclidiana entre tais pontos. Onde $X_1,X_2\in\mathbb{R}^{n}$ 
$$dist(X_1, X_2) = \sqrt[]{\sum_{i=1}^n{(X_1i-X_2i)^{2}}}$$

[Referêcia 3.1](https://pt.wikipedia.org/wiki/Distância_euclidiana#:~:text=Em%20matemática%2C%20distância%20euclidiana%20é,torna-se%20um%20espaço%20métrico.)


## 4. Dilema

    Tal processo se mostrou redundante, visto que realizava diversos passos desnecessários e que muitas vezes prejudicavam uma previsão mais precisa. Para contornar o problema, mostrou-se necessário uma quebra de paradigma. Deixando a teoria de Euclides de lado e partindo para um conceito novo.
---
## 5. Confeção - treinamento
----

### 5.1.1 Função Gamma
 $\Gamma(X)$
- A função Gamma cria um lambda com a média das normas dos vetores mais próximos a um novo ponto quaisquer.
---

Sejam
$\lambda_{ij}, X = (x_1, \dots, x_2)\in\mathbb{R^{n}}$, $a, b = 0$ 

E a função gama definida por
$$\Gamma(X) = \lambda=(\sum_{i=1}^{n}{(\sum_{j=1}^{len(X_i)}a+1, b+x_{ij})})$$
    Onde a soma de a só será realizada se e somente se Xij for diferente de zero, caso contrário, nada será feito.
    A função Gamma fica completa com um passo a mais em cada interação do somatório primário. Passo a de uma forma nominal significa a o somátorio da média das normas presentes no X. Ou seja:

$$ \lambda = \bigcup_{k=1}^{n}(a_i/b_i) \leftrightarrow b_i \neq 0
$$

$$\dots b_1 = 0 \rightarrow \lambda_i = C
$$

    De uma forma nominal, quando bi for igual a zero, a função acrescenta uma constante qualquer na matriz lambda para que não interfira nas contas seguintes. Tal função foi escrita na linguagem python da seguinte forma:
---
```python
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
```
---
### 5.1.2 Método Theta ($\Theta(column)$)
---
- Método Theta caso seja específicado uma coluna, organiza de forma crescenteos vetores colunas com base na coluna especificada e devolve o 3 matrizes. São elas: [mínimo de todas as espécies separados por caracteristica], [maximo de todas as espécies separados por caracteristica] e [desvio padrão de todas as espécies separados por caracteristicas].

---
        Caso 1: Coluna especificada:

Sejam $\gamma\in[0,1,2 ,3]$ e $\zeta = [X_1, X_2,\dots,X_n]\in \mathbb{R^{n}}$

$$
step = [\bigcup_{i=1}^{n}[\bigcup_{j=1}^{\zeta_i}\zeta_i\leftrightarrow\zeta_i<\zeta_j\in\forall\zeta]]
$$

$$
\Theta(\gamma) = [[\bigcup_{i=1}^{n}min(step_i)], [\bigcup_{i=1}^{n} max(step_i)], [\bigcup_{i=1}^{n}std(step_i)]]
$$
---
        Caso 2: Coluhna não específicada:

Seja $\zeta = [X_1, X_2, \dots, X_n]\in\mathbb{R^{n}}$
$$
\Theta(X) = \zeta = \bigcup_{i=1}^{n}sorted(\zeta_i)
$$

        Onde a função sorted organiza a matriz em ordem crescente.
---

Tal método fica do modo em python:

---
```python
def rearranges(self, command="norm"):
    match command:
        
        case "norm": 
            self.matrix = np.array([sorted(self.matrix[i]) for i in range(len(self.matrix))])

        case _:
            a = np.array([np.array([i.args_num[command] for i in self.matrix[j]]) for j in range(len(self.matrix))])
            return np.array([[np.min(a[i]) for i in range(len(a))], [np.max(a[i]) for i in range(len(a))], [np.std(a[i]) for i in range(len(a))]])
```

---

### 5.1.3 Método Omega $\Omega()$

Sejam 

$\varphi, \lambda, \tau,n,m\in\mathbb{R}$

$\nu,\rho\in\mathbb{R^{n}}$

$\epsilon(x):= \|x\|$

$\mu = (v.a._1,v.a._2,\dots, v.a._m)$

$\sigma := \nu_{\mu \lambda}$


$$
    \psi = \Gamma(\Theta,\bigcup_{j}^{\rho}[\bigcup_{k}^{j}[\sqrt{\sum[\bigcup_{l=0}^{\varphi}(\theta_{\sigma l})^{2}}\leftrightarrow \epsilon(\theta)-\tau <=\epsilon(k) <= \epsilon(\theta)+\tau]]])
$$

$\Omega := min(\psi)$
---
---
- Após o cálculo do psi, Omega busca qual dos componentes de psi é o minimo para a tomada de decisão. Ou seja:$\Omega = min(\psi)$. Com tal função, a máquina tem como acuraccy 95,1% com 33% do banco de dados para treinamento. Se mostrando eficiente com até 1% do banco de dados para treinamento, resultando em 90% de acuraccy.






---
```python
def prev(self):
    counter = 0

    while True:
        self.rearranges()
        teste = self.original_matrix[self.index[counter]]
        counter+=1
        m = ver2([[sum([(teste.lista(k)-j.lista(k))**2 for k in range(len(j.args_num)) if teste.fo()-self.norm_var <= j.fo() <= teste.fo()+self.norm_var])**(1/2) for j in i]for i in self.matrix])
    
        if ((m[0] < m[1]) & (m[0] < m[2])):
            if (teste.species != "versicolor"):
                self.falso_posi +=1
                self.dict["Versicolor"][f"{teste.species}"] += 1
                
            else:                    
                self.matrix[0].append(teste)
                self.dict["Versicolor"][f"{teste.species}"] += 1
                

        if ((m[1] < m[0]) & (m[1] < m[2])):
            if (teste.species != "setosa"): 
                self.falso_posi += 1
                self.dict["Setosa"][f"{teste.species}"] += 1

            else:                    
                self.matrix[1].append(teste)
                self.dict["Setosa"][f"{teste.species}"] += 1
            
        if ((m[2] < m[1]) & (m[2] < m[0])):
            if (teste.species != "virginica"):
                self.falso_posi +=1 
                self.dict["Virginica"][f"{teste.species}"] += 1
                
            else:
                self.matrix[2].append(teste)
                self.dict["Virginica"][f"{teste.species}"] += 1

                
        self.dict["Total"] += 1

        if (counter == len(self.index)):
                self.acuracy =1-(self.falso_posi/len(self.index))
                self.file(f"Teste numero: {self.now}   Falso positivo: {self.falso_posi}   Precisao: {self.acuracy}   tamanho indices:   {len(self.index)}   Total : {counter}   Tempo de execucao unitario: {datetime.now()-self.start}\n\n")


        if (counter == len(self.index)):
            self.acuracy =1-(self.falso_posi/len(self.index))
            self.total_analisado = len(self.index)
            break    
```

---


## 5.1.1 [Matriz de confusão](https://pt.wikipedia.org/wiki/Matriz_de_confusão)
> Resumidamente, matriz de confusão tem como objetivo mostrar os erros e acertos da inteligência artificial. A mesma mostra os acertos do processo na diagonal principal.
---

    Com testes relaizados com apenas 33% do banco de dados para treinamento, um padrão de erro se mostrou. Em grande parte dos casos, a matriz de confusão resultante tomava tal formato:

---
---
* Teste com 33% do banco de dados para treinamento (acuracy = 95%)

Espécie: | Virginica | Versicolor | Setosa| Total
-------|------|------|---------|----
**Virginica** | 12 | 1 | 0 | 13
**Versicolor** | 1 | 13 | 0 | 14
**Setosa** | 0 | 0 | 14| 14
|||||41

---

---
---
* Teste com 10% do banco de dados para treinamento (acuracy = 92,307%)

Espécie: | Virginica | Versicolor | Setosa| Total
-------|------|------|---------|----
**Virginica** | 5 | 1 | 0 | 6
**Versicolor** | 0 | 3 | 0 | 3
**Setosa** | 0 | 0 | 4| 4
|||||13

---
Gráficos com vetores apenas da base de dados (10% para treinamento)

---
[2D](https://uploaddeimagens.com.br/imagens/4YxHqbc)

[3D](https://uploaddeimagens.com.br/imagens/lyRzAuU)

---
Gráficos com 1000 vetores aleatórios (10% para treinamento)

---
[2D](https://uploaddeimagens.com.br/imagens/nLGEGlI)

[3D](https://uploaddeimagens.com.br/imagens/ElFX2Ag)

---
---
* Teste com menos de 10% do banco de dados para treinamento (acuracy = 90%)

Espécie: | Virginica | Versicolor | Setosa| Total
-------|------|------|---------|----
**Virginica** | 4 | 1 | 0 | 5
**Versicolor** | 0 | 2 | 0 | 2
**Setosa** | 0 | 0 | 3| 3
|||||10

---



