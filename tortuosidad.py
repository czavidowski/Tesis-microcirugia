import numpy as np
def calcular_tortuosidad(puntos):

    puntos = np.array(puntos, dtype=float)

    if len(puntos) < 2:
        raise ValueError("Se necesitan al menos dos puntos")

    # DISTANCIA POLIGONAL
    distancias = []

    for i in range(len(puntos) - 1):
        d = np.linalg.norm(puntos[i + 1] - puntos[i])
        distancias.append(d)

    d_poligonal = np.sum(distancias)

    # RECTA IDEAL POR PCA
    media = np.mean(puntos, axis=0)
    puntos_centrados = puntos - media

    cov = np.cov(puntos_centrados.T)

    autovalores, autovectores = np.linalg.eigh(cov)

    indice_mayor = np.argmax(autovalores)
    direccion_principal = autovectores[:, indice_mayor]

    # PROYECTAR LOS PUNTOS SOBRE LA DIRECCIÓN PRINCIPAL
    proyecciones = puntos_centrados @ direccion_principal

    d_lineal_pca = np.max(proyecciones) - np.min(proyecciones)

    if d_lineal_pca == 0:
        raise ValueError("No se puede calcular la distancia PCA")

    # TORTUOSIDAD PCA
    tortuosidad_pca = d_poligonal / d_lineal_pca

    return tortuosidad_pca
