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

    # -----------------------------------------------------
    # 1. MÁSCARA HSV
    # -----------------------------------------------------

    hsv = cv2.cvtColor(
        imagen_bgr,
        cv2.COLOR_BGR2HSV
    )

    mascara_hsv = cv2.inRange(
        hsv,
        AZUL_BAJO,
        AZUL_ALTO
    )


    # -----------------------------------------------------
    # 2. COMPROBAR QUE REALMENTE PREDOMINE EL AZUL
    # -----------------------------------------------------

    B, G, R = cv2.split(imagen_bgr)

    B = B.astype(np.int16)
    G = G.astype(np.int16)
    R = R.astype(np.int16)

    predominio_azul = (
        (B > G + 15) &
        (B > R + 15)
    )


    # -----------------------------------------------------
    # 3. EVITAR ZONAS MUY OSCURAS
    # -----------------------------------------------------

    valor = hsv[:, :, 2]

    no_muy_oscuro = valor > 50


    # -----------------------------------------------------
    # 4. COMBINAR CONDICIONES
    # -----------------------------------------------------

    mascara = (
        (mascara_hsv > 0) &
        predominio_azul &
        no_muy_oscuro
    )

    mascara = (
        mascara.astype(np.uint8) * 255
    )


    # -----------------------------------------------------
    # 5. LIMPIEZA MORFOLÓGICA
    # -----------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    mascara_limpia = cv2.morphologyEx(
        mascara,
        cv2.MORPH_OPEN,
        kernel
    )

    mascara_limpia = cv2.morphologyEx(
        mascara_limpia,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mascara_limpia