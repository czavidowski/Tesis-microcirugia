
import cv2
from pathlib import Path

import procesamiento
from io_imagenes import obtener_imagenes
from segmentacion import segmentar_hilo
from deteccion_nudos import detectar_nudos
from visualizacion import mostrar_nudos_finales, mostrar_tortuosidad
from tortuosidad import calcular_tortuosidad

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

    mascara_inicial = segmentar_hilo(imagen)
    # ---------------------------------------------
    # ORIENTACIÓN POR HOTELLING
    # ---------------------------------------------
    angulo_rotacion = procesamiento.calcular_angulo_hotelling(
        mascara_inicial
    )
    # ---------------------------------------------
    # ROTAR IMAGEN
    # ---------------------------------------------
    imagen_rotada, M = procesamiento.rotar_sin_recortar(
        imagen,
        angulo_rotacion,
        (255, 255, 255)
    )
    # ---------------------------------------------
    # VOLVER A SEGMENTAR LA IMAGEN ROTADA
    # ---------------------------------------------
    mascara_hilo = segmentar_hilo(
        imagen_rotada
    )
    # ---------------------------------------------
    # DETECCIÓN DE NUDOS
    # ---------------------------------------------
    resultado = detectar_nudos(
        mascara_hilo
    )

    cantidad_nudos = len(resultado["nudos"])

    # ---------------------------------------------
    # OBTENER CENTROIDES FINALES
    # ---------------------------------------------
    centroides = procesamiento.obtener_centroides(
        resultado["nudos"]
    )
    # ---------------------------------------------
    # CALCULAR TORTUOSIDAD
    # ---------------------------------------------
    if len(centroides) >= 2:

        tortuosidad_pca = calcular_tortuosidad(centroides)
        print(f"Tortuosidad PCA: {tortuosidad_pca:.4f}")
        mostrar_tortuosidad(imagen_rotada, centroides, Path(ruta_imagen).name)

    else:
        print("No hay suficientes nudos para calcular tortuosidad")

    # ---------------------------------------------
    # NOMBRE DE LA IMAGEN
    # ---------------------------------------------
    nombre_imagen = Path(ruta_imagen).name

    print(f"{nombre_imagen} -> {cantidad_nudos} nudos")

    # ---------------------------------------------
    # MOSTRAR RESULTADO FINAL
    # ---------------------------------------------
    mostrar_nudos_finales(
        imagen_rotada,
        resultado["labels_cand"],
        resultado["candidatos"],
        resultado["nudos"],
        nombre_imagen
    )

# =========================================================
# CALIBRACIÓN
# =========================================================
px_por_mm, mm_por_px = procesamiento.calibrar_imagen(
    "escala.jpeg",
    20
)

print(f"Píxeles por mm: {px_por_mm:.2f}")
print(f"Milímetros por píxel: {mm_por_px:.4f}")
