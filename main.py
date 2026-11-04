import cv2
from pathlib import Path
from collections import defaultdict
from tkinter import Tk, messagebox, simpledialog

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
# AGRUPAR IMÁGENES POR ESTUDIANTE Y MOMENTO
# =========================================================

imagenes_por_estudiante = defaultdict(
    lambda: {
        "inicial": [],
        "final": [],
        "extra": []
    }
)

for ruta_imagen in rutas_imagenes:

    ruta = Path(ruta_imagen)

    # Ejemplo:
    # imagenes/puntos_iniciales/estudiante1/foto1.jpg
    #
    # ruta.parent.name        -> estudiante1
    # ruta.parent.parent.name -> puntos_iniciales

    estudiante = ruta.parent.name
    carpeta_momento = ruta.parent.parent.name.lower()

    if "inicial" in carpeta_momento:
        momento = "inicial"

    elif "final" in carpeta_momento:
        momento = "final"

    elif "extra" in carpeta_momento:
        momento = "extra"

    else:
        continue

    imagenes_por_estudiante[
        estudiante
    ][momento].append(
        ruta_imagen
    )


# =========================================================
# ORDENAR LAS IMÁGENES DE CADA ESTUDIANTE
# =========================================================

for estudiante in imagenes_por_estudiante:

    for momento in [
        "inicial",
        "final",
        "extra"
    ]:

        imagenes_por_estudiante[
            estudiante
        ][momento].sort()


# =========================================================
# CALIBRACIÓN
# =========================================================

px_por_mm, mm_por_px = procesamiento.calibrar_imagen(
    "escala.jpeg",
    20
)


# =========================================================
# PREPARAR TKINTER
# =========================================================

root = Tk()
root.withdraw()
numero_estudiante = simpledialog.askinteger(
    "Seleccionar estudiante",
    "¿Qué número de estudiante querés ver?"
)
if numero_estudiante is None:
    root.destroy()
    exit()

estudiante_buscado = f"estudiante_{numero_estudiante}"

# =========================================================
# PROCESAR ESTUDIANTE POR ESTUDIANTE
# =========================================================

for estudiante in sorted(imagenes_por_estudiante.keys()):

    if estudiante != estudiante_buscado:
        continue

    momentos = imagenes_por_estudiante[
        estudiante
    ]

    # Acá vamos guardando solamente las tortuosidades
    # ya calculadas para este estudiante.
    tortuosidades = {
        "inicial": [],
        "final": [],
        "extra": []
    }


    # =====================================================
    # RECORRER LOS MOMENTOS DEL ESTUDIANTE
    # =====================================================

    for momento in [
        "inicial",
        "final",
        "extra"
    ]:

        rutas = momentos[momento]

        # Si este estudiante no tiene ese momento,
        # simplemente seguimos con el próximo.
        if len(rutas) == 0:
            continue

        print(
            f"\nProcesando "
            f"{estudiante} - {momento}"
        )


        # =================================================
        # PROCESAR LAS IMÁGENES DE ESE MOMENTO
        # =================================================

        for ruta_imagen in rutas:

            imagen = cv2.imread(
                ruta_imagen
            )

            if imagen is None:

                print(
                    f"No se pudo cargar: "
                    f"{ruta_imagen}"
                )

                continue


            # =============================================
            # 1. SEGMENTAR LA IMAGEN ORIGINAL
            # =============================================

            mascara_hilo = segmentar_hilo(
                imagen
            )


            # =============================================
            # 2. DETECTAR NUDOS EN LA IMAGEN ORIGINAL
            # =============================================

            resultado = detectar_nudos(
                mascara_hilo
            )

            cantidad_nudos = len(
                resultado["nudos"]
            )


            # =============================================
            # 3. CALCULAR ORIENTACIÓN POR HOTELLING
            # =============================================

            angulo_rotacion = (
                procesamiento.calcular_angulo_hotelling(
                    mascara_hilo
                )
            )


            # =============================================
            # 4. ROTAR LA IMAGEN
            # =============================================

            imagen_rotada, M = (
                procesamiento.rotar_sin_recortar(
                    imagen,
                    angulo_rotacion,
                    (255, 255, 255)
                )
            )


            # =============================================
            # 5. TRANSFORMAR CENTROIDES
            # =============================================

            centroides_rotados = []

            for nudo in resultado["nudos"]:

                centroide_rotado = (
                    procesamiento.transformar_punto(
                        nudo["centroide"],
                        M
                    )
                )

                nudo[
                    "centroide_rotado"
                ] = centroide_rotado

                centroides_rotados.append(
                    centroide_rotado
                )


            # =============================================
            # 6. DISTANCIA ENTRE EXTREMOS
            # =============================================

            distancia_mm = distancia_extremos(
                centroides_rotados,
                mm_por_px
            )


            # =============================================
            # 7. DISTANCIA POLIGONAL
            # =============================================

            distancia_poligonal_mm = (
                distancia_poligonal(
                    centroides_rotados,
                    mm_por_px
                )
            )


            # =============================================
            # 8. TORTUOSIDAD
            # =============================================

            tortuosidad = calcular_tortuosidad(
                distancia_poligonal_mm,
                distancia_mm
            )


            # =============================================
            # GUARDAR TORTUOSIDAD PARA EL PROMEDIO
            # =============================================

            if tortuosidad is not None:

                # La tortuosidad matemática sigue siendo,
                # por ejemplo, 1.0023.
                #
                # Para mostrarla usamos:
                # 0.23 %

                tortuosidad_pct = (
                    tortuosidad - 1
                ) * 100

                tortuosidades[
                    momento
                ].append(
                    tortuosidad_pct
                )


            # =============================================
            # 9. EQUIDISTANCIA
            # =============================================

            (
                distancias_mm,
                distancia_ideal,
                desviaciones
            ) = desviacion_equidistancia(
                centroides_rotados,
                mm_por_px
            )


            # =============================================
            # 10. NOMBRE DE LA IMAGEN
            # =============================================

            nombre_imagen = Path(
                ruta_imagen
            ).name


            # =============================================
            # 11. LÍNEA ENTRE EXTREMOS
            # =============================================

            imagen_con_linea = (
                dibujar_linea_extremos(
                    imagen_rotada,
                    centroides_rotados
                )
            )


            # =============================================
            # 12. LÍNEA POLIGONAL
            # =============================================

            imagen_con_lineas = (
                dibujar_linea_poligonal(
                    imagen_con_linea,
                    centroides_rotados
                )
            )


            # =============================================
            # 13. PANEL DE RESULTADOS
            # =============================================

            imagen_con_resultados = (
                dibujar_resultados(
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
            )


            # =============================================
            # 14. MOSTRAR RESULTADO
            # =============================================

            mostrar_nudos_rotados(
                imagen_con_resultados,
                resultado["labels_cand"],
                resultado["candidatos"],
                resultado["nudos"],
                M,
                f"{estudiante} - {momento} - {nombre_imagen}"
            )


    # =====================================================
    # TERMINARON TODAS LAS IMÁGENES DEL ESTUDIANTE
    # =====================================================

    promedios = {}


    # =====================================================
    # CALCULAR PROMEDIOS
    # =====================================================

    for momento in [
        "inicial",
        "final",
        "extra"
    ]:

        valores = tortuosidades[
            momento
        ]

        if len(valores) > 0:

            promedios[momento] = (
                sum(valores)
                / len(valores)
            )


    # =====================================================
    # ARMAR RESUMEN DEL ESTUDIANTE
    # =====================================================

    lineas_resumen = [
        f"ESTUDIANTE: {estudiante}",
        ""
    ]


    if "inicial" in promedios:

        lineas_resumen.append(
            "Tortuosidad promedio inicial: "
            f"{promedios['inicial']:.2f} %"
        )


    if "final" in promedios:

        lineas_resumen.append(
            "Tortuosidad promedio final: "
            f"{promedios['final']:.2f} %"
        )


    if "extra" in promedios:

        lineas_resumen.append(
            "Tortuosidad promedio extra: "
            f"{promedios['extra']:.2f} %"
        )


    # =====================================================
    # MEJORA INICIAL -> FINAL
    # =====================================================

    if (
        "inicial" in promedios
        and "final" in promedios
    ):

        # =====================================================
        # MEJORA RELATIVA INICIAL -> FINAL
        # =====================================================

        if promedios["inicial"] > 0:

            mejora_porcentual = (
                                        (
                                                promedios["inicial"]
                                                - promedios["final"]
                                        )
                                        / promedios["inicial"]
                                ) * 100

            lineas_resumen.append("")

            if mejora_porcentual > 0:

                lineas_resumen.append(
                    "Reducción de la tortuosidad: "
                    f"{mejora_porcentual:.2f} %"
                )

            elif mejora_porcentual < 0:

                lineas_resumen.append(
                    "Aumento de la tortuosidad: "
                    f"{abs(mejora_porcentual):.2f} %"
                )

            else:

                lineas_resumen.append(
                    "La tortuosidad no presentó cambios."
                )


    # =====================================================
    # MEJORA FINAL -> EXTRA
    # =====================================================

    if (
        "final" in promedios
        and "extra" in promedios
    ):

        mejora_final_extra = (
            promedios["final"]
            - promedios["extra"]
        )

        lineas_resumen.append(
            "Mejora final -> extra: "
            f"{mejora_final_extra:.2f} "
            "puntos porcentuales"
        )


    # =====================================================
    # MEJORA TOTAL INICIAL -> EXTRA
    # =====================================================

    if (
        "inicial" in promedios
        and "extra" in promedios
    ):

        mejora_total = (
            promedios["inicial"]
            - promedios["extra"]
        )

        lineas_resumen.append(
            "Mejora total inicial -> extra: "
            f"{mejora_total:.2f} "
            "puntos porcentuales"
        )


    resumen = "\n".join(
        lineas_resumen
    )


    # =====================================================
    # MOSTRAR CUADRO DE RESUMEN
    # =====================================================

    messagebox.showinfo(
        f"Resumen - {estudiante}",
        resumen
    )


    # =====================================================
    # PREGUNTAR SI QUIERE VER EL PRÓXIMO ESTUDIANTE
    # =====================================================

    continuar = messagebox.askyesno(
        "Continuar",
        "¿Desea ver el próximo estudiante?"
    )

    if not continuar:
        break


# =========================================================
# CERRAR TKINTER
# =========================================================

root.destroy()