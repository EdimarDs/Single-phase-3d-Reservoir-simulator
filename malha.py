import numpy as np
from m3d import read_matrix_3d

def generate_reservoir_properties():

    permeabilidade = np.zeros((5, 10, 10))
    # Leitura dos arquivos de texto
    permeabilidade = read_matrix_3d('./Dados/fperm.txt')  # Permeabilidade m3d
    viscosidade = read_matrix_3d('./Dados/fvisc.txt')  # Viscosidade m3d
    porosidade = read_matrix_3d('./Dados/fporo.txt')  # Porosidade m3d

    cps = 3.5e-6  # Compressibilidade do sistema

    return porosidade, viscosidade, permeabilidade, cps