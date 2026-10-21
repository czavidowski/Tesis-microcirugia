import matplotlib.pyplot as plt
from PIL import Image
import math
import cv2
import numpy as np

def calibrar_imagen(ruta_imagen, distancia_real_mm):
    imagen = Image.open(ruta_imagen)
    # Mostrar imagen
    plt.figure()
    plt.imshow(imagen)
    plt.title("Hace clic en dos marcas de la regla")
    # Marcar dos puntos con el mouse
    puntos = plt.ginput(2)
    # Separar coordenadas
    x1, y1 = puntos[0]
    x2, y2 = puntos[1]
    # Calcular distancia entre los puntos en pixeles
    distancia_px = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    # Calibracion
    px_por_mm = distancia_px / distancia_real_mm
    mm_por_px = distancia_real_mm / distancia_px
    # Dibujar los puntos y la linea
    plt.plot([x1, x2], [y1, y2], 'ro')
    plt.plot([x1, x2], [y1, y2], 'r-')
    plt.show()
    # Mostrar resultados
    print(f"Distancia medida en la imagen: {distancia_px:.2f} pixeles")
    print(f"Calibracion: {px_por_mm:.4f} pixeles/mm")
    print(f"Calibracion: {mm_por_px:.6f} mm/pixel")
    # Devolver valores
    return px_por_mm, mm_por_px

def rotar_sin_recortar(imagen, angulo, border_value):
    (h, w) = imagen.shape[:2]
    (cx, cy) = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D((cx, cy), angulo, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    nuevo_w = int((h * sin) + (w * cos))
    nuevo_h = int((h * cos) + (w * sin))
    M[0, 2] += (nuevo_w / 2) - cx
    M[1, 2] += (nuevo_h / 2) - cy
    rotada = cv2.warpAffine(
        imagen,
        M,
        (nuevo_w, nuevo_h),
        borderValue=border_value
    )
    return rotada, M

def rotar_mismo_tamanio(imagen, angulo, border_value):
    h, w = imagen.shape[:2]
    centro = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(
        centro,
        angulo,
        1.0
    )
    rotada = cv2.warpAffine(
        imagen,
        M,
        (w, h),
        borderValue=border_value
    )
    return rotada, M


def obtener_centroides(nudos):

    centroides = []

    for nudo in nudos:
        centroides.append(nudo["centroide"])

    return centroides


def calcular_angulo_hotelling(mascara):

    # Coordenadas de todos los píxeles del hilo
    y, x = np.where(mascara > 0)

    if len(x) < 2:
        raise ValueError("No hay suficientes píxeles para calcular la orientación")

    puntos = np.column_stack((x, y)).astype(float)

    # Centrar los puntos
    media = np.mean(puntos, axis=0)
    puntos_centrados = puntos - media

    # Matriz de covarianza
    cov = np.cov(puntos_centrados.T)

    # Autovalores y autovectores
    autovalores, autovectores = np.linalg.eigh(cov)

    # Dirección principal = autovector del mayor autovalor
    direccion_principal = autovectores[
        :,
        np.argmax(autovalores)
    ]

    vx, vy = direccion_principal

    # Ángulo de la dirección principal
    angulo = np.degrees(
        np.arctan2(vy, vx)
    )

    # El autovector puede apuntar para cualquiera de los dos lados.
    # Lo normalizamos entre 0 y 180°.
    angulo = angulo % 180

    # Queremos que la sutura quede vertical (90°)
    angulo_rotacion = angulo - 90

    return angulo_rotacion
def transformar_punto(punto, M):

    x, y = punto

    nuevo_x = (
        M[0, 0] * x +
        M[0, 1] * y +
        M[0, 2]
    )

    nuevo_y = (
        M[1, 0] * x +
        M[1, 1] * y +
        M[1, 2]
    )

    return float(nuevo_x), float(nuevo_y)