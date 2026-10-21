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
    cv2.line(imagen_resultado, punto_inicial, punto_final, color_linea, grosor)

    # Dibujar extremos
    cv2.circle(imagen_resultado, punto_inicial, 5, (0, 0, 255), -1)  # rojo
    cv2.circle(imagen_resultado, punto_final, 5, (255, 0, 0), -1)    # azul

    return imagen_resultado

def mostrar_nudos_rotados(
    imagen_rotada,
    labels_cand,
    candidatos,
    nudos,
    M,
    nombre_imagen=""
):

    imagen_rgb = cv2.cvtColor(
        imagen_rotada,
        cv2.COLOR_BGR2RGB
    )

    alto_rotado, ancho_rotado = imagen_rotada.shape[:2]

    # Candidatos que quedaron como representantes
    # de los nudos finales
    representantes = {
        nudo["rep"]
        for nudo in nudos
    }

    # -----------------------------------------------------
    # ROTAR LAS MÁSCARAS DE LOS NUDOS YA DETECTADOS
    # -----------------------------------------------------

    for candidato in candidatos:

        if candidato["id"] not in representantes:
            continue

        # Máscara del candidato en la imagen ORIGINAL
        mascara_candidato = np.where(
            labels_cand == candidato["label"],
            255,
            0
        ).astype(np.uint8)

        # Llevar esa máscara a la imagen rotada
        mascara_rotada = cv2.warpAffine(
            mascara_candidato,
            M,
            (ancho_rotado, alto_rotado),
            flags=cv2.INTER_NEAREST,
            borderValue=0
        )

        dibujar_contorno(
            imagen_rgb,
            mascara_rotada,
            (0, 200, 0),
            3
        )

    # -----------------------------------------------------
    # MOSTRAR
    # -----------------------------------------------------

    plt.figure(
        num=nombre_imagen,
        figsize=(10, 7)
    )

    plt.imshow(imagen_rgb)

    # Números y centroides ya transformados
    for numero, nudo in enumerate(
        nudos,
        start=1
    ):

        x, y = nudo["centroide_rotado"]

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