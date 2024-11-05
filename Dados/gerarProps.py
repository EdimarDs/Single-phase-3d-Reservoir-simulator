import numpy as np

# Definindo as dimensões da matriz
dimensao = (3, 3, 4)  # 5 planos, 10 linhas, 10 colunas

# Gerando valores aleatórios de permeabilidade (em mD) entre 50 e 500
# Aqui, estamos utilizando uma faixa de valores comumente usada para rochas reservatório
permeabilidade = np.random.uniform(0.25, 0.35, dimensao)

# Salvando a matriz em um arquivo .txt
with open('matriz_permeabilidade.txt', 'w') as f:
    for plano in permeabilidade:
        for linha in plano:
            # Convertendo a linha em string e separando os elementos por espaço
            f.write(" ".join(map(str, linha)) + "\n")
        f.write("\n")  # Adiciona uma linha em branco para separar os planos
