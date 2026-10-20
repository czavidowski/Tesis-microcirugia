import cv2
import numpy as np


# =========================================================
# PARÁMETROS
# =========================================================

TAM_KERNEL_NUDOS = 15

# Por ahora no eliminamos componentes chicas automáticamente
FRACCION_RUIDO = 0

# Fusionar si la brecha geodésica es menor o igual
# al diámetro típico de los candidatos
FACTOR_FUSION = 1.0

# Máximo recorrido a explorar para la distancia geodésica
TOPE_MEDICION = 10


# =========================================================
# PASO 1
# SEPARAR COMPONENTES CONEXAS DEL HILO ORIGINAL
# =========================================================

def separar_hilos(mascara):

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mascara,
        connectivity=8
    )

    if n <= 1:
        return mascara, labels, stats

    areas = stats[1:, cv2.CC_STAT_AREA]

    if FRACCION_RUIDO > 0:
        area_min = FRACCION_RUIDO * areas.max()
    else:
        area_min = 0

    mascara_limpia = np.zeros_like(mascara)

    for i in range(1, n):

        area = stats[i, cv2.CC_STAT_AREA]

        if area >= area_min:
            mascara_limpia[labels == i] = 255

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mascara_limpia,
        connectivity=8
    )

    return mascara_limpia, labels, stats


# =========================================================
# PASO 2
# GENERAR CANDIDATOS A NUDO
# =========================================================

def detectar_candidatos(mascara, labels_hilo):

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (TAM_KERNEL_NUDOS, TAM_KERNEL_NUDOS)
    )

    mascara_nudos = cv2.morphologyEx(
        mascara,
        cv2.MORPH_OPEN,
        kernel
    )

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mascara_nudos,
        connectivity=8
    )

    candidatos = []

    for i in range(1, n):

        area = int(
            stats[i, cv2.CC_STAT_AREA]
        )

        ys, xs = np.nonzero(
            labels == i
        )

        if len(xs) == 0:
            continue

        # Buscar a qué componente del hilo original pertenece
        labels_coincidentes = labels_hilo[
            ys,
            xs
        ]

        labels_coincidentes = labels_coincidentes[
            labels_coincidentes > 0
        ]

        if len(labels_coincidentes) == 0:
            continue

        valores, cantidades = np.unique(
            labels_coincidentes,
            return_counts=True
        )

        hilo = int(
            valores[
                np.argmax(cantidades)
            ]
        )

        # Diámetro equivalente del candidato
        diametro = float(
            np.sqrt(
                4 * area / np.pi
            )
        )

        candidatos.append(
            {
                "id": len(candidatos),
                "label": i,
                "area": area,
                "diametro": diametro,
                "centroide": (
                    float(centroids[i][0]),
                    float(centroids[i][1])
                ),
                "hilo": hilo
            }
        )

    return mascara_nudos, labels, candidatos


# =========================================================
# PASO 3
# DISTANCIA GEODÉSICA ENTRE DOS CANDIDATOS
# =========================================================

def brecha_geodesica(
    hilo_roi,
    candidato_a,
    candidato_b,
    tope
):

    kernel = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    frente = candidato_a.copy()

    for paso in range(
        1,
        tope + 1
    ):

        dilatado = cv2.dilate(
            frente,
            kernel
        )

        # Solo se puede avanzar por dentro del hilo
        nuevo = cv2.bitwise_and(
            dilatado,
            hilo_roi
        )

        # Ver si ya tocamos al otro candidato
        contacto = cv2.bitwise_and(
            nuevo,
            candidato_b
        )

        if cv2.countNonZero(contacto) > 0:
            return paso

        crecimiento = cv2.subtract(
            nuevo,
            frente
        )

        if cv2.countNonZero(crecimiento) == 0:
            return None

        frente = nuevo

    return None


# =========================================================
# PASO 4
# MEDIR BRECHAS ENTRE CANDIDATOS DEL MISMO HILO
# =========================================================

def medir_brechas(
    candidatos,
    labels_cand,
    labels_hilo,
    stats_hilo,
    tope
):

    por_hilo = {}

    for candidato in candidatos:

        hilo = candidato["hilo"]

        por_hilo.setdefault(
            hilo,
            []
        ).append(
            candidato
        )

    brechas = {}

    for hilo, lista in por_hilo.items():

        if len(lista) < 2:
            continue

        x = stats_hilo[
            hilo,
            cv2.CC_STAT_LEFT
        ]

        y = stats_hilo[
            hilo,
            cv2.CC_STAT_TOP
        ]

        w = stats_hilo[
            hilo,
            cv2.CC_STAT_WIDTH
        ]

        h = stats_hilo[
            hilo,
            cv2.CC_STAT_HEIGHT
        ]

        sl = (
            slice(y, y + h),
            slice(x, x + w)
        )

        hilo_roi = np.where(
            labels_hilo[sl] == hilo,
            255,
            0
        ).astype(
            np.uint8
        )

        for i in range(
            len(lista)
        ):

            for j in range(
                i + 1,
                len(lista)
            ):

                candidato_a = lista[i]
                candidato_b = lista[j]

                mascara_a = np.where(
                    labels_cand[sl]
                    == candidato_a["label"],
                    255,
                    0
                ).astype(
                    np.uint8
                )

                mascara_b = np.where(
                    labels_cand[sl]
                    == candidato_b["label"],
                    255,
                    0
                ).astype(
                    np.uint8
                )

                distancia = brecha_geodesica(
                    hilo_roi,
                    mascara_a,
                    mascara_b,
                    tope
                )

                brechas[
                    (
                        candidato_a["id"],
                        candidato_b["id"]
                    )
                ] = distancia

    return brechas


# =========================================================
# PASO 5
# AGRUPAR CANDIDATOS MUY CERCANOS
# =========================================================

def agrupar_candidatos(
    candidatos,
    brechas,
    umbral
):

    padre = list(
        range(
            len(candidatos)
        )
    )

    def raiz(i):

        while padre[i] != i:

            padre[i] = padre[
                padre[i]
            ]

            i = padre[i]

        return i

    for (i, j), distancia in brechas.items():

        if (
            distancia is not None
            and
            distancia <= umbral
        ):

            padre[
                raiz(i)
            ] = raiz(j)

    grupos = {}

    for candidato in candidatos:

        r = raiz(
            candidato["id"]
        )

        grupos.setdefault(
            r,
            []
        ).append(
            candidato
        )

    nudos = []

    for miembros in grupos.values():

        # Por ahora elegimos como representante
        # el candidato de mayor área
        representante = max(
            miembros,
            key=lambda c: c["area"]
        )

        nudos.append(
            {
                "rep": representante["id"],
                "centroide": representante["centroide"],
                "hilo": representante["hilo"],
                "miembros": [
                    c["id"]
                    for c in miembros
                ]
            }
        )

    return nudos


# =========================================================
# FUNCIÓN PRINCIPAL DE DETECCIÓN
# =========================================================

def detectar_nudos(mascara):

    """
    Recibe:
        mascara binaria del hilo

    Devuelve:
        mascara_nudos
        labels_hilo
        labels_cand
        candidatos
        brechas
        nudos
        diametro_tipico
        umbral_fusion
    """

    # Componentes del hilo original
    mascara, labels_hilo, stats_hilo = separar_hilos(
        mascara
    )

    # Candidatos a nudo
    mascara_nudos, labels_cand, candidatos = detectar_candidatos(
        mascara,
        labels_hilo
    )

    if len(candidatos) == 0:

        return {
            "mascara_nudos": mascara_nudos,
            "labels_hilo": labels_hilo,
            "labels_cand": labels_cand,
            "candidatos": [],
            "brechas": {},
            "nudos": [],
            "diametro_tipico": None,
            "umbral_fusion": None
        }

    # Calcular tamaño típico de los candidatos
    diametros = [
        c["diametro"]
        for c in candidatos
    ]

    diametro_tipico = float(
        np.median(
            diametros
        )
    )

    # Definir umbral de fusión
    umbral_fusion = (
        FACTOR_FUSION *
        diametro_tipico
    )

    # Tope máximo para buscar caminos
    tope = int(
        np.ceil(
            max(
                TOPE_MEDICION * diametro_tipico,
                umbral_fusion
            )
        )
    )

    # Medir distancias geodésicas
    brechas = medir_brechas(
        candidatos,
        labels_cand,
        labels_hilo,
        stats_hilo,
        tope
    )

    # Agrupar candidatos
    nudos = agrupar_candidatos(
        candidatos,
        brechas,
        umbral_fusion
    )

    return {
        "mascara_nudos": mascara_nudos,
        "labels_hilo": labels_hilo,
        "labels_cand": labels_cand,
        "candidatos": candidatos,
        "brechas": brechas,
        "nudos": nudos,
        "diametro_tipico": diametro_tipico,
        "umbral_fusion": umbral_fusion
    }