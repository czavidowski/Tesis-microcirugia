import cv2
import numpy as np
# =========================================================
# PARÁMETROS DE SEGMENTACIÓN
# =========================================================
AZUL_BAJO = np.array([90, 60, 40])
AZUL_ALTO = np.array([140, 255, 255])
# =========================================================
# SEGMENTAR EL HILO AZUL
# =========================================================
def segmentar_hilo(imagen_bgr):
    # Convertir de BGR a HSV
    hsv = cv2.cvtColor(
        imagen_bgr,
        cv2.COLOR_BGR2HSV
    )
    # Segmentar el color azul
    mascara = cv2.inRange(
        hsv,
        AZUL_BAJO,
        AZUL_ALTO
    )
    # Kernel pequeño para limpiar la máscara
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )
    # Eliminar pequeños puntos aislados
    mascara_limpia = cv2.morphologyEx(
        mascara,
        cv2.MORPH_OPEN,
        kernel
    )
    # Cerrar pequeños huecos
    mascara_limpia = cv2.morphologyEx(
        mascara_limpia,
        cv2.MORPH_CLOSE,
        kernel
    )
    return mascara_limpia