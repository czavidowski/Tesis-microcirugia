import cv2
from pathlib import Path

import procesamiento
from io_imagenes import obtener_imagenes
from segmentacion import segmentar_hilo
from deteccion_nudos import detectar_nudos
from visualizacion import mostrar_nudos_rotados, dibujar_linea_extremos
from tortuosidad import distancia_extremos


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


    # ---------------------------------------------
    # 1. SEGMENTAR IMAGEN ORIGINAL
    # ---------------------------------------------

    mascara_hilo = segmentar_hilo(
        imagen
    )


    # ---------------------------------------------
    # 2. DETECTAR NUDOS EN LA IMAGEN ORIGINAL
    # ---------------------------------------------

    resultado = detectar_nudos(
        mascara_hilo
    )

    cantidad_nudos = len(
        resultado["nudos"]
    )


    # ---------------------------------------------
    # 3. CALCULAR ORIENTACIÓN POR HOTELLING
    # ---------------------------------------------

    angulo_rotacion = procesamiento.calcular_angulo_hotelling(
        mascara_hilo
    )


    # ---------------------------------------------
    # 4. ROTAR LA IMAGEN
    # ---------------------------------------------

    imagen_rotada, M = procesamiento.rotar_sin_recortar(
        imagen,
        angulo_rotacion,
        (255, 255, 255)
    )


    # ---------------------------------------------
    # 5. TRANSFORMAR CENTROIDES A LA IMAGEN ROTADA
    # ---------------------------------------------

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


    # ---------------------------------------------
    # 6. DISTANCIA ENTRE PRIMER Y ÚLTIMO NUDO
    # ---------------------------------------------

    distancia_mm = distancia_extremos(
        centroides_rotados,
        mm_por_px
    )


    # ---------------------------------------------
    # 7. NOMBRE DE LA IMAGEN
    # ---------------------------------------------

    nombre_imagen = Path(
        ruta_imagen
    ).name


    print(
        f"{nombre_imagen} -> "
        f"{cantidad_nudos} nudos"
    )

    if distancia_mm is not None:

        print(
            f"Distancia entre extremos: "
            f"{distancia_mm:.2f} mm"
        )

    else:

        print(
            "No hay suficientes centroides "
            "para calcular la distancia"
        )


    # ---------------------------------------------
    # 8. DIBUJAR LÍNEA SOBRE IMAGEN ROTADA
    # ---------------------------------------------

    imagen_con_linea = dibujar_linea_extremos(
        imagen_rotada,
        centroides_rotados
    )


    # ---------------------------------------------
    # 9. MOSTRAR IMAGEN ROTADA CON LOS NUDOS
    #    DETECTADOS EN LA IMAGEN ORIGINAL
    # ---------------------------------------------

    mostrar_nudos_rotados(
        imagen_con_linea,
        resultado["labels_cand"],
        resultado["candidatos"],
        resultado["nudos"],
        M,
        nombre_imagen
    )