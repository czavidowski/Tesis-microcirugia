import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# DIBUJAR CONTORNO
# =========================================================
def dibujar_contorno(
    imagen_rgb,
    mascara_binaria,
    color,
    grosor=2
):
    contornos, _ = cv2.findContours(
        mascara_binaria,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )
    cv2.drawContours(
        imagen_rgb,
        contornos,
        -1,
        color,
        grosor
    )
# =========================================================
# MOSTRAR SEGMENTACIÓN Y CANDIDATOS
# =========================================================
def mostrar_proceso(
    imagen_bgr,
    mascara_hilo,
    mascara_nudos
):
    plt.figure(
        figsize=(15, 5)
    )
    # Imagen original
    plt.subplot(
        1,
        3,
        1
    )
    plt.imshow(
        cv2.cvtColor(
            imagen_bgr,
            cv2.COLOR_BGR2RGB
        )
    )
    plt.title(
        "Imagen original"
    )
    plt.axis(
        "off"
    )
    # Máscara del hilo
    plt.subplot(
        1,
        3,
        2
    )
    plt.imshow(
        mascara_hilo,
        cmap="gray"
    )
    plt.title(
        "Hilo segmentado"
    )
    plt.axis(
        "off"
    )
    # Máscara de candidatos
    plt.subplot(
        1,
        3,
        3
    )
    plt.imshow(
        mascara_nudos,
        cmap="gray"
    )
    plt.title(
        "Candidatos después de apertura 15x15"
    )
    plt.axis(
        "off"
    )
    plt.tight_layout()
    plt.show()
# =========================================================
# MOSTRAR COMPONENTES CONEXAS Y CANDIDATOS
# =========================================================
def mostrar_candidatos(
    imagen_bgr,
    labels_hilo,
    labels_cand,
    candidatos
):
    plt.figure(
        figsize=(14, 6)
    )
    # ---------------------------------------------
    # COMPONENTES DEL HILO
    # ---------------------------------------------
    plt.subplot(
        1,
        2,
        1
    )
    hilos = np.ma.masked_where(
        labels_hilo == 0,
        labels_hilo
    )
    plt.imshow(
        hilos,
        cmap="tab20"
    )
    plt.title(
        "Componentes conexas del hilo original"
    )
    plt.axis(
        "off"
    )
    # ---------------------------------------------
    # CANDIDATOS
    # ---------------------------------------------
    plt.subplot(
        1,
        2,
        2
    )
    vis = cv2.cvtColor(
        imagen_bgr,
        cv2.COLOR_BGR2RGB
    )
    for candidato in candidatos:
        mascara_candidato = np.where(
            labels_cand
            == candidato["label"],
            255,
            0
        ).astype(
            np.uint8
        )
        dibujar_contorno(
            vis,
            mascara_candidato,
            (255, 0, 0),
            2
        )
    plt.imshow(
        vis
    )
    for candidato in candidatos:
        x, y = candidato[
            "centroide"
        ]
        plt.text(
            x,
            y,
            f"C{candidato['id']}\nH{candidato['hilo']}",
            color="red",
            fontsize=9,
            ha="center",
            va="center",
            weight="bold"
        )
    plt.title(
        "Candidatos y componente a la que pertenecen"
    )
    plt.axis(
        "off"
    )
    plt.tight_layout()
    plt.show()
# =========================================================
# MOSTRAR RESULTADO FINAL
# =========================================================
def mostrar_nudos_finales(
    imagen_bgr,
    labels_cand,
    candidatos,
    nudos,
    nombre_imagen=""
):
    vis = cv2.cvtColor(
        imagen_bgr,
        cv2.COLOR_BGR2RGB
    )
    representantes = {
        nudo["rep"]
        for nudo in nudos
    }
    for candidato in candidatos:
        if candidato["id"] not in representantes:
            continue
        mascara_candidato = np.where(
            labels_cand == candidato["label"],
            255,
            0
        ).astype(np.uint8)
        dibujar_contorno(
            vis,
            mascara_candidato,
            (0, 200, 0),
            3
        )
    plt.figure(
        num=nombre_imagen,
        figsize=(10, 7)
    )
    plt.imshow(vis)
    for numero, nudo in enumerate(
        nudos,
        start=1
    ):
        x, y = nudo["centroide"]
        plt.plot(
            x,
            y,
            "y+",
            markersize=14,
            mew=2
        )
        plt.text(
            x,
            y - 25,
            str(numero),
            color="yellow",
            fontsize=14,
            ha="center",
            weight="bold"
        )
    plt.title(
        f"{nombre_imagen} - Nudos detectados: {len(nudos)}"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.show()
# =========================================================
# MOSTRAR INFORMACIÓN EN CONSOLA
# =========================================================
def imprimir_resultados(resultado):
    candidatos = resultado[
        "candidatos"
    ]
    brechas = resultado[
        "brechas"
    ]
    nudos = resultado[
        "nudos"
    ]
    diametro_tipico = resultado[
        "diametro_tipico"
    ]
    umbral_fusion = resultado[
        "umbral_fusion"
    ]
    print(
        "=" * 60
    )
    print(
        "Cantidad de candidatos:",
        len(candidatos)
    )
    if diametro_tipico is not None:
        print(
            f"Diámetro típico: "
            f"{diametro_tipico:.2f} px"
        )
        print(
            f"Umbral de fusión: "
            f"{umbral_fusion:.2f} px"
        )
    print(
        "\n--- CANDIDATOS ---"
    )
    for candidato in candidatos:
        x, y = candidato[
            "centroide"
        ]
        print(
            f"C{candidato['id']}: "
            f"hilo={candidato['hilo']}, "
            f"area={candidato['area']}, "
            f"diametro={candidato['diametro']:.2f}, "
            f"centro=({x:.1f}, {y:.1f})"
        )
    print(
        "\n--- BRECHAS GEODESICAS ---"
    )
    if len(brechas) == 0:
        print(
            "No hay candidatos del mismo hilo para comparar."
        )
    for (i, j), distancia in brechas.items():
        if distancia is None:
            print(
                f"C{i} - C{j}: "
                f"sin conexión dentro del rango medido"
            )
        else:
            if distancia <= umbral_fusion:
                decision = "FUSIONAR"
            else:
                decision = "SEPARADOS"
            print(
                f"C{i} - C{j}: "
                f"{distancia} px "
                f"-> {decision}"
            )
    print(
        "\nNUDOS DETECTADOS:",
        len(nudos)
    )
    for numero, nudo in enumerate(
        nudos,
        start=1
    ):
        x, y = nudo[
            "centroide"
        ]
        print(
            f"Nudo {numero}: "
            f"centro=({x:.1f}, {y:.1f}), "
            f"candidatos={nudo['miembros']}"
        )


# =========================================================
# VISUALIZAR linea recta
# =========================================================
import cv2

# =========================================================
# VISUALIZAR linea recta
# =========================================================
import cv2

def dibujar_linea_extremos(imagen, centroides, color_linea=(0, 255, 0), grosor=2):

    imagen_resultado = imagen.copy()

    if len(centroides) < 2:
        return imagen_resultado

    # Primer y último punto
    punto_inicial = tuple(map(int, centroides[0]))
    punto_final = tuple(map(int, centroides[-1]))

    # Dibujar línea
    cv2.line(imagen_resultado, punto_inicial, punto_final, (0,0,0), 3)

    # Dibujar extremos
    cv2.circle(imagen_resultado, punto_inicial, 5, (0, 0, 255), -1)  # rojo
    cv2.circle(imagen_resultado, punto_final, 5, (255, 0, 0), -1)    # azul

    return imagen_resultado


def dibujar_linea_poligonal(imagen, centroides, color_linea=(0, 0, 255), grosor=3):

    imagen_resultado = imagen.copy()

    if len(centroides) != 5:
        return imagen_resultado

    # Dibujar las líneas entre puntos consecutivos
    for i in range(len(centroides) - 1):

        punto_inicial = tuple(map(int, centroides[i]))
        punto_final = tuple(map(int, centroides[i + 1]))

        cv2.line(
            imagen_resultado,
            punto_inicial,
            punto_final,
            color_linea,
            grosor
        )

    # Dibujar los 5 centroides
    for centroide in centroides:

        punto = tuple(map(int, centroide))

        cv2.circle(imagen_resultado, punto, 5, (0, 0, 255), -1)

    return imagen_resultado
def mostrar_nudos_rotados(
    imagen_rotada,
    labels_cand,
    candidatos,
    nudos,
    M,
    nombre_imagen=""
):

    # =====================================================
    # SEPARAR FOTO Y PANEL DE RESULTADOS
    # =====================================================

    ancho_panel = 600

    alto_total, ancho_total = imagen_rotada.shape[:2]

    ancho_foto = ancho_total - ancho_panel

    # Parte izquierda = foto
    foto_bgr = imagen_rotada[
        :,
        :ancho_foto
    ]

    # Parte derecha = panel
    panel_bgr = imagen_rotada[
        :,
        ancho_foto:
    ]

    foto_rgb = cv2.cvtColor(
        foto_bgr,
        cv2.COLOR_BGR2RGB
    )

    panel_rgb = cv2.cvtColor(
        panel_bgr,
        cv2.COLOR_BGR2RGB
    )


    # =====================================================
    # DIBUJAR CONTORNOS DE LOS NUDOS
    # =====================================================

    representantes = {
        nudo["rep"]
        for nudo in nudos
    }

    for candidato in candidatos:

        if candidato["id"] not in representantes:
            continue

        mascara_candidato = np.where(
            labels_cand == candidato["label"],
            255,
            0
        ).astype(np.uint8)

        # Rotamos la máscara al mismo tamaño que la FOTO,
        # no incluyendo el panel lateral
        mascara_rotada = cv2.warpAffine(
            mascara_candidato,
            M,
            (ancho_foto, alto_total),
            flags=cv2.INTER_NEAREST,
            borderValue=0
        )

        dibujar_contorno(
            foto_rgb,
            mascara_rotada,
            (0, 200, 0),
            3
        )


    # =====================================================
    # CALCULAR ZOOM AUTOMÁTICO SEGÚN LOS NUDOS
    # =====================================================

    xs = []
    ys = []

    for nudo in nudos:

        x, y = nudo["centroide_rotado"]

        xs.append(x)
        ys.append(y)

    if len(xs) > 0:

        margen_x = 220
        margen_y = 180

        x_min = max(
            0,
            min(xs) - margen_x
        )

        x_max = min(
            ancho_foto,
            max(xs) + margen_x
        )

        y_min = max(
            0,
            min(ys) - margen_y
        )

        y_max = min(
            alto_total,
            max(ys) + margen_y
        )

    else:

        x_min = 0
        x_max = ancho_foto

        y_min = 0
        y_max = alto_total


    # =====================================================
    # RECORTAR ESPACIO VACÍO DEL PANEL
    # =====================================================

    # El panel tiene fondo aproximadamente 245.
    # Buscamos hasta dónde realmente hay texto.
    diferencia = np.any(
        panel_bgr < 235,
        axis=2
    )

    filas_con_texto = np.where(
        np.any(diferencia, axis=1)
    )[0]

    if len(filas_con_texto) > 0:

        ultima_fila = min(
            alto_total,
            filas_con_texto[-1] + 80
        )

        panel_rgb = panel_rgb[
            :ultima_fila,
            :
        ]


    # =====================================================
    # CREAR FIGURA
    # =====================================================

    fig, (ax_imagen, ax_panel) = plt.subplots(
        1,
        2,
        figsize=(16, 9),
        gridspec_kw={
            "width_ratios": [3.2, 1.4]
        }
    )


    # =====================================================
    # MOSTRAR IMAGEN CON ZOOM
    # =====================================================

    ax_imagen.imshow(
        foto_rgb
    )

    # Aplicar zoom
    ax_imagen.set_xlim(
        x_min,
        x_max
    )

    # En imágenes el eje Y crece hacia abajo
    ax_imagen.set_ylim(
        y_max,
        y_min
    )


    # =====================================================
    # MOSTRAR CENTROIDES Y NÚMEROS
    # =====================================================

    for numero, nudo in enumerate(
        nudos,
        start=1
    ):

        x, y = nudo["centroide_rotado"]

        ax_imagen.plot(
            x,
            y,
            "y+",
            markersize=18,
            mew=3
        )

        ax_imagen.text(
            x,
            y - 20,
            str(numero),
            color="yellow",
            fontsize=18,
            ha="center",
            weight="bold"
        )

    ax_imagen.axis(
        "off"
    )


    # =====================================================
    # MOSTRAR PANEL DE RESULTADOS
    # =====================================================

    ax_panel.imshow(
        panel_rgb
    )

    ax_panel.axis(
        "off"
    )


    # =====================================================
    # TÍTULO
    # =====================================================

    fig.suptitle(
        f"{nombre_imagen} - Nudos detectados: {len(nudos)}",
        fontsize=18
    )

    plt.tight_layout()
    plt.show()
def dibujar_resultados(
    imagen,
    cantidad_nudos,
    distancia_mm,
    distancia_poligonal_mm,
    tortuosidad,
    distancia_ideal,
    distancias_mm,
    desviaciones,
    px_por_mm,
    mm_por_px
):

    # -----------------------------------------------------
    # TORTUOSIDAD EN PORCENTAJE
    # -----------------------------------------------------

    if tortuosidad is not None:
        tortuosidad_pct = (tortuosidad - 1) * 100
    else:
        tortuosidad_pct = None


    # -----------------------------------------------------
    # CREAR PANEL LATERAL
    # -----------------------------------------------------

    alto, ancho = imagen.shape[:2]

    ancho_panel = 600

    panel = np.ones(
        (alto, ancho_panel, 3),
        dtype=np.uint8
    ) * 245


    # -----------------------------------------------------
    # ARMAR TEXTOS
    # -----------------------------------------------------

    lineas = []

    lineas.append(
        f"Nudos detectados: {cantidad_nudos}"
    )

    lineas.append("")

    lineas.append(
        f"Calibracion:"
    )

    lineas.append(
        f"{px_por_mm:.2f} px/mm"
    )

    lineas.append(
        f"{mm_por_px:.4f} mm/px"
    )

    lineas.append("")

    if distancia_mm is not None:

        lineas.append(
            f"Distancia entre extremos:"
        )

        lineas.append(
            f"{distancia_mm:.4f} mm"
        )


    lineas.append("")


    if distancia_poligonal_mm is not None:

        lineas.append(
            f"Distancia poligonal:"
        )

        lineas.append(
            f"{distancia_poligonal_mm:.4f} mm"
        )


    lineas.append("")


    if tortuosidad_pct is not None:

        lineas.append(
            f"Tortuosidad:"
        )

        lineas.append(
            f"{tortuosidad_pct:.2f} %"
        )


    lineas.append("")


    if distancia_ideal is not None:

        lineas.append(
            f"Distancia ideal entre puntos:"
        )

        lineas.append(
            f"{distancia_ideal:.4f} mm"
        )


    lineas.append("")


    # -----------------------------------------------------
    # DISTANCIAS ENTRE NUDOS
    # -----------------------------------------------------

    if desviaciones is not None:

        lineas.append(
            "Separacion entre puntos:"
        )

        lineas.append("")

        for i, desviacion in enumerate(
            desviaciones
        ):

            distancia = distancias_mm[i]

            if desviacion > 0:

                texto = (
                    f"P{i+1}-P{i+2}: "
                    f"{distancia:.3f} mm "
                    f"(+{desviacion:.3f})"
                )

            elif desviacion < 0:

                texto = (
                    f"P{i+1}-P{i+2}: "
                    f"{distancia:.3f} mm "
                    f"(-{abs(desviacion):.3f})"
                )

            else:

                texto = (
                    f"P{i+1}-P{i+2}: "
                    f"{distancia:.3f} mm "
                    f"(ideal)"
                )

            lineas.append(texto)


    # -----------------------------------------------------
    # DIBUJAR TEXTO
    # -----------------------------------------------------

    x = 35
    y = 60

    separacion = 42

    for texto in lineas:

        if texto == "":
            y += 20
            continue

        cv2.putText(
            panel,
            texto,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (30, 30, 30),
            2,
            cv2.LINE_AA
        )

        y += separacion


    # -----------------------------------------------------
    # UNIR PANEL + IMAGEN
    # -----------------------------------------------------

    imagen_final = np.hstack(
        (imagen, panel)
    )

    return imagen_final