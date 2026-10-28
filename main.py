
import cv2
from pathlib import Path

import procesamiento
from io_imagenes import obtener_imagenes
from segmentacion import segmentar_hilo
from deteccion_nudos import detectar_nudos

from visualizacion import (
    mostrar_nudos_rotados,
    dibujar_linea_extremos,
    dibujar_linea_poligonal
)

from distancia import (
    distancia_extremos,
    distancia_poligonal,
    calcular_tortuosidad
)


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

    if distancia_mm is not None:
        print(
            f"Distancia entre extremos: "
            f"{distancia_mm:.4f} mm"
        )
    else:
        print(
            "No se pudo calcular "
            "la distancia entre extremos"
        )

    # =====================================================
    # 7. DISTANCIA POLIGONAL
    # =====================================================

    distancia_poligonal_mm = distancia_poligonal(
        centroides_rotados,
        mm_por_px
    )

    if distancia_poligonal_mm is not None:
        print(
            f"Distancia poligonal: "
            f"{distancia_poligonal_mm:.4f} mm"
        )
    else:
        print(
            "No se pudo calcular "
            "la distancia poligonal"
        )

    # =====================================================
    # 8. CALCULAR TORTUOSIDAD
    # =====================================================

    tortuosidad = calcular_tortuosidad(
        distancia_poligonal_mm,
        distancia_mm
    )

    if tortuosidad is not None:
        print(f"Tortuosidad: {tortuosidad:.4f}")
    else:
        print("No se pudo calcular la tortuosidad")

    # =====================================================
    # 9. NOMBRE DE LA IMAGEN
    # =====================================================

    nombre_imagen = Path(ruta_imagen).name

    print(
        f"{nombre_imagen} -> "
        f"{cantidad_nudos} nudos"
    )

    # =====================================================
    # 10. DIBUJAR LÍNEA ENTRE EXTREMOS
    # =====================================================

    imagen_con_linea = dibujar_linea_extremos(
        imagen_rotada,
        centroides_rotados
    )

    # =====================================================
    # 11. DIBUJAR LÍNEA POLIGONAL
    # =====================================================

    imagen_con_lineas = dibujar_linea_poligonal(
        imagen_con_linea,
        centroides_rotados
    )

    # =====================================================
    # 12. MOSTRAR RESULTADO FINAL ROTADO
    # =====================================================

    mostrar_nudos_rotados(
        imagen_con_lineas,
        resultado["labels_cand"],
        resultado["candidatos"],
        resultado["nudos"],
        M,
        nombre_imagen
    )
