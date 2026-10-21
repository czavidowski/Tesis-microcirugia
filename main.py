
import cv2
from pathlib import Path
import procesamiento
from io_imagenes import obtener_imagenes
from segmentacion import segmentar_hilo
from deteccion_nudos import detectar_nudos
from visualizacion import mostrar_nudos_finales, dibujar_linea_extremos
from distancia import distancia_extremos,distancia_poligonal, calcular_tortuosidad

# =========================================================
# CARGAR TODAS LAS IMÁGENES
# =========================================================
rutas_imagenes = obtener_imagenes("imagenes")

# =========================================================
# CALIBRACIÓN
# =========================================================
px_por_mm, mm_por_px = procesamiento.calibrar_imagen(
    "escala.jpeg",
    20
)

print(f"Píxeles por mm: {px_por_mm:.2f}")
print(f"Milímetros por píxel: {mm_por_px:.4f}")

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
    # OBTENER CENTROIDES ORDENADOS
    # ---------------------------------------------
    centroides = procesamiento.obtener_centroides(
        resultado["nudos"]
    )

    # ---------------------------------------------
    # DISTANCIA ENTRE PRIMER Y ÚLTIMO NUDO
    # ---------------------------------------------
    distancia_mm = distancia_extremos(
        centroides,
        mm_por_px
    )
    # ---------------------------------------------
    # DISTANCIA POLIGONAL
    # ---------------------------------------------
    distancia_poligonal_mm = distancia_poligonal(
        centroides,
        mm_por_px
    )
    if distancia_poligonal_mm is not None:
        print(f"Distancia poligonal: {distancia_poligonal_mm:.2f} mm")
    else:
        print("No se detectaron exactamente 5 nudos")

    # ---------------------------------------------
    # CALCULAR TORTUOSIDAD
    # ---------------------------------------------
    tortuosidad = calcular_tortuosidad(
        distancia_poligonal_mm,
        distancia_mm
    )

    if tortuosidad is not None:
        print(f"Tortuosidad: {tortuosidad:.4f}")
    else:
        print("No se pudo calcular la tortuosidad")

    # ---------------------------------------------
    # NOMBRE DE LA IMAGEN
    # ---------------------------------------------
    nombre_imagen = Path(ruta_imagen).name

    print(f"{nombre_imagen} -> {cantidad_nudos} nudos")

    if distancia_mm is not None:
        print(f"Distancia entre extremos: {distancia_mm:.2f} mm")
    else:
        print("No hay suficientes centroides para calcular la distancia")

    # ---------------------------------------------
    # DIBUJAR LÍNEA SOBRE IMAGEN ROTADA
    # ---------------------------------------------
    imagen_con_linea = dibujar_linea_extremos(
        imagen_rotada,
        centroides
    )

    # ---------------------------------------------
    # MOSTRAR RESULTADO FINAL
    # ---------------------------------------------
    mostrar_nudos_finales(
        imagen_con_linea,
        resultado["labels_cand"],
        resultado["candidatos"],
        resultado["nudos"],
        nombre_imagen
    )

