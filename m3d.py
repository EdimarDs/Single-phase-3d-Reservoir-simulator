import numpy as np

def read_matrix_3d(filename):
    with open(filename, 'r') as file:
        data = file.read().strip()
        
    # Separar os planos usando duas quebras de linha
    planes = data.split('\n\n')
    
    matrix_3d = []
    
    for plane in planes:
        # Ler cada plano e transformar em uma lista de listas
        rows = plane.splitlines()
        matrix_2d = [list(map(float, row.split())) for row in rows]
        matrix_3d.append(matrix_2d)

    return np.array(matrix_3d)
