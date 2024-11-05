import numpy as np
import matplotlib.pyplot as plt

# Atualize a assinatura de generate_plots para incluir well_cell

def generate_plots(nx, ny, nz, Lx_cells, Ly_cells, Lz_cells, time_steps, P_results, P_well_over_time, well_cell, q_sc):
    """ Gera gráficos de pressão e vazão ao longo do tempo """
    
    # Slice em z = meio do reservatório
    mid_z = nz // 2

    for t_idx, time in enumerate(time_steps):
        plt.figure(figsize=(10, 6))
        plt.contourf(np.arange(nx), np.arange(ny), P_results[:, :, mid_z, t_idx], cmap='viridis')
        plt.colorbar(label='Pressão (psi)')
        plt.title(f'Pressão no meio do reservatório em t = {time} dias')
        plt.xlabel('Células (x)')
        plt.ylabel('Células (y)')
        plt.show()

    # Gráfico da pressão no poço ao longo do tempo
    plt.figure(figsize=(8, 6))
    plt.plot(time_steps, P_well_over_time, 'o-', label='Pressão no Poço')
    plt.xlabel('Tempo (dias)')
    plt.ylabel('Pressão (psi)')
    plt.title('Pressão no poço ao longo do tempo')
    plt.legend()
    plt.show()