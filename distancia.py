import numpy as np

def distancia_extremos(centroides, mm_por_px):

    if len(centroides) < 2:
        return None

    # Primer y último centroide
    x1, y1 = centroides[0]
    x2, y2 = centroides[-1]

    # Distancia euclídea en píxeles
    distancia_px = ((x2 - x1)**2 + (y2 - y1)**2)**0.5

    # Conversión a milímetros
    distancia_mm = distancia_px * mm_por_px

    return distancia_mm


def distancia_poligonal(centroides, mm_por_px):

    if len(centroides) != 5:
        return None

    distancia_total_px = 0

    for i in range(len(centroides) - 1):

        x1, y1 = centroides[i]
        x2, y2 = centroides[i + 1]

        distancia_px = ((x2 - x1)**2 + (y2 - y1)**2)**0.5

        distancia_total_px += distancia_px

    distancia_total_mm = distancia_total_px * mm_por_px

    return distancia_total_mm


def calcular_tortuosidad(distancia_poligonal_mm, distancia_extremos_mm):

    if distancia_poligonal_mm is None or distancia_extremos_mm is None:
        return None

    if distancia_extremos_mm == 0:
        return None

    tortuosidad = distancia_poligonal_mm / distancia_extremos_mm

    return tortuosidad
