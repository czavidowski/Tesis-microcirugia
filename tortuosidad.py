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

