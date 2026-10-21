import cv2
from pathlib import Path
from io_imagenes import obtener_imagenes
from segmentacion import segmentar_hilo
from deteccion_nudos import detectar_nudos
from visualizacion import mostrar_nudos_finales
# =========================================================
# CARGAR TODAS LAS IMÁGENES
# =========================================================
rutas_imagenes = obtener_imagenes("imagenes")
# =========================================================
# PROCESAR CADA IMAGEN
# =========================================================
for ruta_imagen in rutas_imagenes:
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        print(f"No se pudo cargar: {ruta_imagen}")
        continue
    # ---------------------------------------------
    # SEGMENTACIÓN
    # ---------------------------------------------
    mascara_hilo = segmentar_hilo(imagen)
    # ---------------------------------------------
    # DETECCIÓN DE NUDOS
    # ---------------------------------------------
    resultado = detectar_nudos(mascara_hilo)
    cantidad_nudos = len(resultado["nudos"])
    print(
        f"{ruta_imagen} -> "
        f"{cantidad_nudos} nudos"
    )
    # ---------------------------------------------
    # NOMBRE DE LA IMAGEN
    # ---------------------------------------------
    nombre_imagen = Path(ruta_imagen).name
    # ---------------------------------------------
    # MOSTRAR RESULTADO FINAL
    # ---------------------------------------------
    mostrar_nudos_finales(
        imagen,
        resultado["labels_cand"],
        resultado["candidatos"],
        resultado["nudos"],
        nombre_imagen
    )