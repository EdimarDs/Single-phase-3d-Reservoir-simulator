def ponde_media(L1, L2, P1, P2):
    P = (P1 * L1 + P2 * L2) / (L1 + L2)
    return P

def calculate_transmissibilities(D1, D2, Cell_centri, Cell_more, p, v, Bo, indice, n):

    if indice < n - 1:
        perm = ponde_media(Cell_centri/2,Cell_more/2, p[1], p[2])
        mi = ponde_media(Cell_centri/2,Cell_more/2, v[1], v[2])
        D3 = Cell_centri/2 + Cell_more/2
        T_morehalf = 1.127 * (D1 * D2 * perm) / (mi * Bo * D3)
    else:
        T_morehalf = 0

    return T_morehalf