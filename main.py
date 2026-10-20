import cv2

from segmentacion import segmentar_hilo
from deteccion_nudos import detectar_nudos
from visualizacion import mostrar_nudos_finales
# =========================================================
# IMAGEN A ANALIZAR
# =========================================================
ruta_imagen = "imagenes/imagen3.jpg"
# =========================================================
# CARGAR IMAGEN
# =========================================================
imagen = cv2.imread(ruta_imagen)
if imagen is None:
    raise FileNotFoundError(
        f"No se pudo cargar la imagen: {ruta_imagen}"
    )
# =========================================================
# SEGMENTACIÓN
# =========================================================
mascara_hilo = segmentar_hilo(
    imagen
)
# =========================================================
# DETECCIÓN DE NUDOS
# =========================================================
resultado = detectar_nudos(
    mascara_hilo
)
# =========================================================
# RESULTADO
# =========================================================
cantidad_nudos = len(
    resultado["nudos"]
)
print(
    f"Nudos detectados: {cantidad_nudos}"
)
mostrar_nudos_finales(
    imagen,
    resultado["labels_cand"],
    resultado["candidatos"],
    resultado["nudos"]
)
