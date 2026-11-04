import cv2
from pathlib import Path

import procesamiento

from io_imagenes import obtener_imagenes
from segmentacion import segmentar_hilo
from deteccion_nudos import detectar_nudos
from visualizacion import *
from distancia import *


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


# =========================================================
# PROCESAR CADA IMAGEN
# =========================================================

for ruta_imagen in rutas_imagenes:

    imagen = cv2.imread(ruta_imagen)

    if imagen is None:
        print(f"No se pudo cargar: {ruta_imagen}")
        continue


    # =====================================================
    # 1. SEGMENTAR LA IMAGEN ORIGINAL
    # =====================================================

    mascara_hilo = segmentar_hilo(
        imagen
    )


    # =====================================================
    # 2. DETECTAR NUDOS EN LA IMAGEN ORIGINAL
    # =====================================================

    resultado = detectar_nudos(
        mascara_hilo
    )

    cantidad_nudos = len(
        resultado["nudos"]
    )


    # =====================================================
    # 3. CALCULAR ORIENTACIÓN POR HOTELLING
    # =====================================================

    angulo_rotacion = procesamiento.calcular_angulo_hotelling(
        mascara_hilo
    )


    # =====================================================
    # 4. ROTAR LA IMAGEN
    # =====================================================

    imagen_rotada, M = procesamiento.rotar_sin_recortar(
        imagen,
        angulo_rotacion,
        (255, 255, 255)
    )


    # =====================================================
    # 5. TRANSFORMAR LOS CENTROIDES YA DETECTADOS
    # =====================================================

    centroides_rotados = []

    for nudo in resultado["nudos"]:

        centroide_rotado = procesamiento.transformar_punto(
            nudo["centroide"],
            M
        )

        nudo["centroide_rotado"] = centroide_rotado

        centroides_rotados.append(
            centroide_rotado
        )


    # =====================================================
    # 6. DISTANCIA ENTRE PRIMER Y ÚLTIMO NUDO
    # =====================================================

    distancia_mm = distancia_extremos(
        centroides_rotados,
        mm_por_px
    )


    # =====================================================
    # 7. DISTANCIA POLIGONAL
    # =====================================================

    distancia_poligonal_mm = distancia_poligonal(
        centroides_rotados,
        mm_por_px
    )


    # =====================================================
    # 8. TORTUOSIDAD
    # =====================================================

    tortuosidad = calcular_tortuosidad(
        distancia_poligonal_mm,
        distancia_mm
    )


    # =====================================================
    # 9. EQUIDISTANCIA
    # =====================================================

    distancias_mm, distancia_ideal, desviaciones = (
        desviacion_equidistancia(
            centroides_rotados,
            mm_por_px
        )
    )


    # =====================================================
    # 10. NOMBRE DE LA IMAGEN
    # =====================================================

    nombre_imagen = Path(
        ruta_imagen
    ).name


    # =====================================================
    # 11. DIBUJAR LÍNEA ENTRE EXTREMOS
    # =====================================================

    imagen_con_linea = dibujar_linea_extremos(
        imagen_rotada,
        centroides_rotados
    )


    # =====================================================
    # 12. DIBUJAR LÍNEA POLIGONAL
    # =====================================================

    imagen_con_lineas = dibujar_linea_poligonal(
        imagen_con_linea,
        centroides_rotados
    )


    # =====================================================
    # 13. AGREGAR RESULTADOS A LA IMAGEN
    # =====================================================

    imagen_con_resultados = dibujar_resultados(
        imagen_con_lineas,
        cantidad_nudos,
        distancia_mm,
        distancia_poligonal_mm,
        tortuosidad,
        distancia_ideal,
        distancias_mm,
        desviaciones,
        px_por_mm,
        mm_por_px
    )


    # =====================================================
    # 14. MOSTRAR NUDOS SOBRE LA IMAGEN ROTADA
    # =====================================================

    mostrar_nudos_rotados(
        imagen_con_resultados,
        resultado["labels_cand"],
        resultado["candidatos"],
        resultado["nudos"],
        M,
        nombre_imagen
    )