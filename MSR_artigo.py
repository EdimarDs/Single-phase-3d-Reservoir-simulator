import numpy as np
from graphs import generate_plots
from malha import generate_reservoir_properties
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from calculate_transmiss import calculate_transmissibilities
from method_jacobi import jacobi_method

def main():
    # Inicialização dos parâmetros do problema
    nx, ny, nz, Lx_cells, Ly_cells, Lz_cells, Lwell, flow_rate, Bo, P_es, dt, time_steps = initialize_parameters()

    # Geração das propriedades do reservatório
    porosidade, viscosidade, permeabilidade, cps = generate_reservoir_properties()

    # Inicialização dos resultados de pressão ao longo do tempo e da vazão (q_sc)
    P_results, P_well_over_time, well_cell, q_sc = initialize_pressure_results(nx, ny, nz, Lx_cells, Ly_cells, Lz_cells, flow_rate, Lwell, time_steps)

    # Cálculo das pressões ao longo do tempo
    P_results = compute_pressures_implicit(nx, ny, nz, Lx_cells, Ly_cells, Lz_cells, porosidade, viscosidade, permeabilidade, Bo, cps, q_sc, P_es, dt, time_steps, P_results)

    # Geração dos gráficos, agora passando também `q_sc` como argumento
    generate_plots(nx, ny, nz, Lx_cells, Ly_cells, Lz_cells, time_steps, P_results, P_results[well_cell], well_cell, q_sc)


def initialize_parameters():
    nx, ny, nz = 3, 3, 3  # Número de células em cada direção
    Lwell = [100, 5000, 5000]  # Localização do poço (x, y, z em ft)
    flow_rate = -100 # Vazão de produção (STB/d)
    Bo = 1.1      # Fator volume de formação do óleo (bbl/STB)
    P_es = 5000   # Pressão inicial (psi)
    dt = 10       # Passo de tempo (dias)
    time_steps = [20, 45, 90, 180, 365, 720] # Tempos de simulação
    
    # Discretização do reservatório com as localizações das faces
    faces_x = np.linspace(0, 100, nx+1)  # Malha regular no eixo x
    faces_y = np.linspace(0, 100, ny+1)  # Malha regular no eixo y
    faces_z = np.linspace(0, 50, nz+1)    # Malha regular no eixo z
    
    Lx_cells = np.diff(faces_x)  # Comprimento das células em x
    Ly_cells = np.diff(faces_y)  # Comprimento das células em y
    Lz_cells = np.diff(faces_z)  # Comprimento das células em z
    
    return nx, ny, nz, Lx_cells, Ly_cells, Lz_cells, Lwell, flow_rate, Bo, P_es, dt, time_steps

def initialize_pressure_results(nx, ny, nz, Lx_cells, Ly_cells, Lz_cells, flow_rate, Lwell, time_steps):
    # Inicializa resultados
    P_results = np.zeros((nx, ny, nz, len(time_steps)))
    P_well_over_time = np.zeros(len(time_steps))
    
    # Localizar célula do poço (agora em 3D)
    well_cell = (
        np.argmin(np.abs(np.cumsum(Lx_cells) - Lwell[0])),  # Índice em x
        np.argmin(np.abs(np.cumsum(Ly_cells) - Lwell[1])),  # Índice em y
        np.argmin(np.abs(np.cumsum(Lz_cells) - Lwell[2]))   # Índice em z
    )

    # Criar matriz de vazão 3D
    q_sc = np.zeros((nx, ny, nz))
    q_sc[well_cell] = flow_rate  # Definir a vazão na célula do poço

    return P_results, P_well_over_time, well_cell, q_sc

def compute_pressures_implicit(nx, ny, nz, Lx_cells, Ly_cells, Lz_cells, porosidade, viscosidade, permeabilidade, Bo, cps, q_sc, P_es, dt, time_steps, P_results):
    for t_idx in range(len(time_steps)):
        tf = time_steps[t_idx]
        P_n = P_es * np.ones((nx, ny, nz))  # Pressão inicial

        for dt_step in np.arange(0, tf, dt):
            ms, vr = build_system(nx, ny, nz, Lx_cells, Ly_cells, Lz_cells, porosidade, viscosidade, permeabilidade, Bo, cps, P_n, q_sc, dt)
            P_n = jacobi_method(ms, vr, nx, ny, nz, tol=1e-10, max_iter=1000) # Atualiza a pressão nas células usando método de Newton
        P_results[:, :, :, t_idx] = P_n  # Armazena a pressão final

    return P_results

# Método de Newton e build_system precisariam ser adaptados para lidar com a matriz tridimensional

def build_transmissibilities(nx, ny, nz, Lx_cells, Ly_cells, Lz_cells, viscosidade, permeabilidade, Bo):
    """ 
    Constrói as transmissibilidades para todas as células no grid 3D.
    """
    Tx = np.zeros((nx+1, ny+1, nz+1))  # Transmissibilidade na direção x
    Ty = np.zeros((nx+1, ny+1, nz+1))  # Transmissibilidade na direção y
    Tz = np.zeros((nx+1, ny+1, nz+1))  # Transmissibilidade na direção z
    
    # Calcular transmissibilidades nas três direções
    for i in range(nx-1):
        for j in range(ny-1):
            for k in range(nz-1):
                dx = Lx_cells[i]
                dy = Ly_cells[j]
                dz = Lz_cells[k]
                if i > 0 and i < nx:
                    px = permeabilidade[i-1][j][k], permeabilidade[i][j][k], permeabilidade[i+1][j][k]
                    vx = [viscosidade[i-1][j][k], viscosidade[i][j][k], viscosidade[i+1][j][k]]
                elif i == 0:
                    px = [0, permeabilidade[i][j][k], permeabilidade[i+1][j][k]]
                    vx = [0, viscosidade[i][j][k], viscosidade[i+1][j][k]]
                else:
                    px = [permeabilidade[i-1][j][k], permeabilidade[i][j][k], 0]
                    vx = [viscosidade[i-1][j][k], viscosidade[i][j][k], 0]

                if j > 0 and j < ny:
                    py = [permeabilidade[i][j-1][k], permeabilidade[i][j][k], permeabilidade[i][j+1][k]]
                    vy = [viscosidade[i][j-1][k], viscosidade[i][j][k], viscosidade[i][j+1][k]]
                elif j == 0:
                    py = [0, permeabilidade[i][j][k], permeabilidade[i][j+1][k]]
                    vy = [0, viscosidade[i][j][k], viscosidade[i][j+1][k]]
                else:
                    py = [permeabilidade[i][j-1][k], permeabilidade[i][j][k], 0]
                    vy = [viscosidade[i][j-1][k], viscosidade[i][j][k], 0]
               
                if k > 0 and k < nz:
                    pz = [permeabilidade[i][j][k-1], permeabilidade[i][j][k], permeabilidade[i][j][k+1]]
                    vz = [viscosidade[i][j][k-1], viscosidade[i][j][k], viscosidade[i][j][k+1]]
                elif k == 0:
                    pz = [0, permeabilidade[i][j][k], permeabilidade[i][j][k+1]]
                    vz = [0, viscosidade[i][j][k], viscosidade[i][j][k+1]]
                else:
                    pz = [permeabilidade[i][j][k-1], permeabilidade[i][j][k], 0]
                    vz = [viscosidade[i][j][k-1], viscosidade[i][j][k], 0]

                if i > 0 and i < nx:
                    Tx[i, j, k] = calculate_transmissibilities(dy, dz, Lx_cells[i], Lx_cells[i+1], px, vx, Bo, i, nx)
                if j > 0 and j < ny:
                    Ty[i, j, k] = calculate_transmissibilities(dx, dz, Ly_cells[j], Ly_cells[j+1], py, vy, Bo, j, ny)
                if k > 0 and k < nz:
                    Tz[i, j, k] = calculate_transmissibilities(dx, dy, Lz_cells[k], Lz_cells[k+1], pz, vz, Bo, k, nz)
    
                # contorno norte e sul
                Tx[0, j, k] = 5000 # direth sul
                Tx[nx, j, k] = 5000 # direth norte
                # contorno leste e oeste
                Ty[i, 0, k] = 5000 # direth oeste
                Ty[i, ny, k] = 5000 # direth leste
                # contorno frente e tras
                Tz[i, j, 0] = 5000 # direth tras
                Tz[i, j, nz] = 5000 # direth frente
    return Tx, Ty, Tz

def build_system(nx, ny, nz, Lx_cells, Ly_cells, Lz_cells, porosidade, viscosidade, permeabilidade, Bo, cps, P_n, q_sc, dt):
    """
    Monta o sistema linear A * P_new = b (ou ms * P_n = vr no código anterior)
    """
    # Inicializa matriz de coeficientes e vetor de resultado
    ms = np.zeros((nx * ny * nz, nx * ny * nz))  # Matriz do sistema
    vr = np.zeros(nx * ny * nz)  # Vetor do lado direito (b)

    Tx, Ty, Tz = build_transmissibilities(nx, ny, nz, Lx_cells, Ly_cells, Lz_cells, viscosidade, permeabilidade, Bo)

    # Percorre todas as células e constrói a matriz A e o vetor b
    for i in range(0, nx):
        for j in range(0, ny):
            for k in range(0, nz):
                idx = k + j * ny + i * ny * nz  # Índice linear da célula (i, j, k) no grid 3D 
                print(i, j, k)
                Vb = Lx_cells[i] * Ly_cells[j] * Lz_cells[k];  
                A = Vb * porosidade[i, j, k] * cps / (5.615 * Bo * dt)
                cell_index = i * ny * nz + j * nz + k
                print(cell_index)
                print('---------------------')
                # Cálculo da matriz A e vetor b para cada célula
                # Diagonal principal:
                ms[cell_index, cell_index] = -(Tx[i+1, j, k] + Tx[i, j, k] +
                               Ty[i, j+1, k] + Ty[i, j, k] +
                               Tz[i, j, k+1] + Tz[i, j, k] + A)
                if i > 0:  # Célula à esquerda
                    ms[cell_index, (i-1) * ny * nz + j * nz + k] = Tx[i, j, k] 
                if i < nx - 1:  # Célula à direita
                    ms[cell_index, (i+1) * ny * nz + j * nz + k] = Tx[i+1, j, k] 
                if j > 0:  # Célula abaixo
                    ms[cell_index, i * ny * nz + (j-1) * nz + k] = Ty[i, j, k] 
                if j < ny - 1:  # Célula acima
                    ms[cell_index, i * ny * nz + (j+1) * nz + k] = Ty[i, j+1, k] 
                if k > 0:  # Célula atrás
                    ms[cell_index, i * ny * nz + j * nz + (k-1)] = Tz[i, j, k]
                if k < nz - 1:  # Célula à frente
                    ms[cell_index, i * ny * nz + j * nz + (k+1)] = Tz[i, j, k+1] 
                vr[cell_index] = -(q_sc[i, j, k] + (Vb * porosidade[i, j, k] * cps / (5.615 * Bo * dt) * P_n[i, j, k]))
    
    # Verificação de elementos numéricos
    # Formatando os valores para que tenham 10 caracteres e 4 casas decimais
    with open('matriz_permeabilidade.txt', 'w') as f:
        for linha in ms:
            # Cada número terá largura fixa de 10 caracteres, com 4 casas decimais
            f.write(" ".join(f"{item:10.4f}" for item in linha) + "\n")
        f.write("\n")

    return ms, vr

if __name__ == "__main__":
    main()