import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# ==========================================================
# 1. SEGMENTAR HILO AZUL
# ==========================================================

def segmentar_hilo(imagen):
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    azul_bajo = np.array([90, 60, 40])
    azul_alto = np.array([140, 255, 255])

    mascara = cv2.inRange(hsv, azul_bajo, azul_alto)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3, 3)
    )

    mascara = cv2.morphologyEx(
        mascara, cv2.MORPH_OPEN, kernel
    )

    mascara = cv2.morphologyEx(
        mascara, cv2.MORPH_CLOSE, kernel
    )

    return mascara


# ==========================================================
# 2. PCA / HOTELLING
# ==========================================================

def calcular_angulo_pca(mascara):
    y, x = np.where(mascara > 0)

    if len(x) < 2:
        raise ValueError("No hay suficientes puntos para PCA")

    puntos = np.column_stack((x, y)).astype(np.float64)

    media = np.mean(puntos, axis=0)
    puntos_centrados = puntos - media

    cov = np.cov(puntos_centrados.T)
    autovalores, autovectores = np.linalg.eigh(cov)

    direccion = autovectores[:, np.argmax(autovalores)]

    angulo = np.degrees(
        np.arctan2(direccion[1], direccion[0])
    )

    return angulo


# ==========================================================
# 3. ROTAR
# ==========================================================

def rotar_mismo_tamanio(imagen, angulo, color_fondo,
                       interpolacion=cv2.INTER_LINEAR):

    alto, ancho = imagen.shape[:2]
    centro = (ancho / 2, alto / 2)

    M = cv2.getRotationMatrix2D(
        centro, angulo, 1.0
    )

    imagen_rotada = cv2.warpAffine(
        imagen,
        M,
        (ancho, alto),
        flags=interpolacion,
        borderValue=color_fondo
    )

    return imagen_rotada, M


# ==========================================================
# 4. DETECCION INICIAL DE CANDIDATOS POR GROSOR
# ==========================================================

def detectar_candidatos(mascara_rotada):

    binaria = (mascara_rotada > 0).astype(np.uint8)

    distancia = cv2.distanceTransform(
        binaria, cv2.DIST_L2, 5
    )

    perfil = np.max(distancia, axis=1)

    perfil_suave = cv2.GaussianBlur(
        perfil.reshape(-1, 1),
        (1, 0),
        sigmaX=0,
        sigmaY=4
    ).ravel()

    valores = perfil_suave[perfil_suave > 0]

    if len(valores) == 0:
        return [], distancia, perfil_suave, np.array([])

    grosor_normal = np.median(valores)

    mad = np.median(
        np.abs(valores - grosor_normal)
    )

    prominencia = max(
        0.30 * grosor_normal,
        2 * mad
    )

    filas = np.where(perfil > 0)[0]
    longitud = filas[-1] - filas[0]

    distancia_minima = max(
        1, int(0.08 * longitud)
    )

    picos, props = find_peaks(
        perfil_suave,
        prominence=prominencia,
        distance=distancia_minima
    )

    centros = []

    for y_pico in picos:

        margen = max(
            5, int(2 * perfil_suave[y_pico])
        )

        y1 = max(0, y_pico - margen)
        y2 = min(mascara_rotada.shape[0], y_pico + margen + 1)

        region = distancia[y1:y2, :]

        y_local, x_centro = np.unravel_index(
            np.argmax(region),
            region.shape
        )

        y_centro = y1 + y_local

        centros.append((int(x_centro), int(y_centro)))

    return centros, distancia, perfil_suave, picos


# ==========================================================
# 5. ANALISIS GEOMETRICO LOCAL
# ==========================================================

def analizar_compactacion(distancia, centros, mascara_rotada):

    alto, ancho = mascara_rotada.shape

    resultados = []
    rois_binarias = []

    for i, (x, y) in enumerate(centros):

        margen = 45

        x1 = max(0, x - margen)
        x2 = min(ancho, x + margen + 1)
        y1 = max(0, y - margen)
        y2 = min(alto, y + margen + 1)

        roi_dist = distancia[y1:y2, x1:x2]

        if roi_dist.size == 0:
            continue

        max_local = np.max(roi_dist)

        if max_local <= 0:
            continue

        # --------------------------------------------------
        # Nos quedamos con el "nucleo grueso" del candidato
        # --------------------------------------------------
        umbral = 0.55 * max_local

        roi_bin = (roi_dist >= umbral).astype(np.uint8) * 255

        # Limpieza suave
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (3, 3)
        )

        roi_bin = cv2.morphologyEx(
            roi_bin, cv2.MORPH_OPEN, kernel
        )

        roi_bin = cv2.morphologyEx(
            roi_bin, cv2.MORPH_CLOSE, kernel
        )

        # --------------------------------------------------
        # Nos quedamos con la componente que contiene
        # el centro del candidato
        # --------------------------------------------------
        cx_local = x - x1
        cy_local = y - y1

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            roi_bin, connectivity=8
        )

        etiqueta_centro = labels[cy_local, cx_local]

        if etiqueta_centro == 0:
            resultados.append({
                "indice": i + 1,
                "centro": (x, y),
                "area": 0,
                "aspecto": np.inf,
                "circularidad": 0,
                "roi_bin": roi_bin
            })
            rois_binarias.append(roi_bin)
            continue

        componente = (labels == etiqueta_centro).astype(np.uint8) * 255
        rois_binarias.append(componente)

        contornos, _ = cv2.findContours(
            componente,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contornos) == 0:
            continue

        contorno = max(contornos, key=cv2.contourArea)

        area = cv2.contourArea(contorno)
        perimetro = cv2.arcLength(contorno, True)

        if perimetro > 0:
            circularidad = 4 * np.pi * area / (perimetro ** 2)
        else:
            circularidad = 0

        if len(contorno) >= 5:
            ellipse = cv2.fitEllipse(contorno)
            (xc, yc), (d1, d2), ang = ellipse

            eje_mayor = max(d1, d2)
            eje_menor = min(d1, d2)

            if eje_menor > 0:
                aspecto = eje_mayor / eje_menor
            else:
                aspecto = np.inf
        else:
            x_box, y_box, w_box, h_box = cv2.boundingRect(contorno)
            if min(w_box, h_box) > 0:
                aspecto = max(w_box, h_box) / min(w_box, h_box)
            else:
                aspecto = np.inf

        resultados.append({
            "indice": i + 1,
            "centro": (x, y),
            "area": area,
            "aspecto": aspecto,
            "circularidad": circularidad,
            "roi_bin": componente
        })

    return resultados, rois_binarias


# ==========================================================
# 6. CLASIFICAR NUDOS
# ==========================================================

def clasificar_candidatos(resultados):
    """
    Regla inicial simple:
    - nudo esperado: mas compacto
    - falso positivo esperado: mas alargado

    Esta regla es experimental.
    """

    seleccionados = []

    print("\n--- METRICAS GEOMETRICAS ---")

    for r in resultados:
        print(
            f"Candidato {r['indice']}: "
            f"area={r['area']:.1f}, "
            f"aspecto={r['aspecto']:.2f}, "
            f"circularidad={r['circularidad']:.3f}"
        )

        # Regla tentativa:
        # - aspecto no muy alto
        # - circularidad razonable
        #
        # Se puede ajustar después con más imágenes.
        es_nudo = (
            r["aspecto"] < 2.8 and
            r["circularidad"] > 0.20
        )

        r["es_nudo"] = es_nudo

        if es_nudo:
            seleccionados.append(r)

    return seleccionados


# ==========================================================
# 7. VISUALIZAR ROIS GEOMETRICAS
# ==========================================================

def mostrar_rois(resultados):

    n = len(resultados)

    plt.figure(figsize=(3 * n, 6))

    for i, r in enumerate(resultados, start=1):

        plt.subplot(2, n, i)
        plt.imshow(r["roi_bin"], cmap="gray")
        plt.title(f"Cand {r['indice']}")
        plt.axis("off")

        plt.subplot(2, n, i + n)
        texto = (
            f"A={r['area']:.0f}\n"
            f"asp={r['aspecto']:.2f}\n"
            f"circ={r['circularidad']:.2f}\n"
            f"nudo={r.get('es_nudo', False)}"
        )

        img_texto = np.ones((120, 120), dtype=np.uint8) * 255
        plt.imshow(img_texto, cmap="gray", vmin=0, vmax=255)
        plt.text(10, 25, texto, fontsize=10, color="black")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# ==========================================================
# 8. MAIN
# ==========================================================

imagen = cv2.imread("imagenes/imagen3.jpg")

if imagen is None:
    raise FileNotFoundError("No se pudo cargar imagen3.jpg")

imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

# Segmentacion
mascara = segmentar_hilo(imagen)

# PCA
angulo = calcular_angulo_pca(mascara)
angulo_rotacion = angulo - 90

print(f"Angulo PCA: {angulo:.2f}")
print(f"Angulo de rotacion: {angulo_rotacion:.2f}")

# Rotacion
imagen_rotada, _ = rotar_mismo_tamanio(
    imagen,
    angulo_rotacion,
    (255, 255, 255)
)

mascara_rotada, _ = rotar_mismo_tamanio(
    mascara,
    angulo_rotacion,
    0,
    interpolacion=cv2.INTER_NEAREST
)

imagen_rotada_rgb = cv2.cvtColor(
    imagen_rotada, cv2.COLOR_BGR2RGB
)

# Deteccion inicial
centros, distancia, perfil, picos = detectar_candidatos(mascara_rotada)

print("\nCantidad inicial de candidatos:", len(centros))
print("Centros iniciales:", centros)

# Analisis geometrico
resultados, rois = analizar_compactacion(
    distancia, centros, mascara_rotada
)

seleccionados = clasificar_candidatos(resultados)

print("\nCantidad final seleccionada:", len(seleccionados))

# ----------------------------------------------------------
# Dibujar todos los candidatos
# ----------------------------------------------------------
resultado_todos = imagen_rotada.copy()

for i, (x, y) in enumerate(centros, start=1):
    cv2.circle(resultado_todos, (x, y), 8, (0, 0, 255), 2)
    cv2.putText(
        resultado_todos,
        str(i),
        (x + 12, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2
    )

resultado_todos_rgb = cv2.cvtColor(
    resultado_todos, cv2.COLOR_BGR2RGB
)

# ----------------------------------------------------------
# Dibujar candidatos finales seleccionados
# ----------------------------------------------------------
resultado_final = imagen_rotada.copy()

for r in seleccionados:
    x, y = r["centro"]
    idx = r["indice"]

    cv2.circle(resultado_final, (x, y), 8, (0, 255, 0), 2)
    cv2.putText(
        resultado_final,
        str(idx),
        (x + 12, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

resultado_final_rgb = cv2.cvtColor(
    resultado_final, cv2.COLOR_BGR2RGB
)

# ==========================================================
# 9. FIGURA GENERAL
# ==========================================================

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(imagen_rgb)
plt.title("Imagen original")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(mascara, cmap="gray")
plt.title("Mascara azul")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(imagen_rotada_rgb)
plt.title("Rotada por PCA")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(distancia, cmap="hot")
plt.title("Mapa de grosor")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.plot(perfil, np.arange(len(perfil)))

for i, y_pico in enumerate(picos, start=1):
    plt.axhline(y_pico, color="red", linestyle="--", alpha=0.5)
    plt.text(max(perfil) * 0.7, y_pico, str(i), color="red")

plt.gca().invert_yaxis()
plt.xlabel("Radio local [px]")
plt.ylabel("Posicion vertical [px]")
plt.title("Perfil de grosor")

plt.subplot(2, 3, 6)
plt.imshow(resultado_todos_rgb)
plt.title(f"Candidatos iniciales: {len(centros)}")
plt.axis("off")

plt.tight_layout()
plt.show()

# ==========================================================
# 10. FIGURA FINAL
# ==========================================================

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(resultado_todos_rgb)
plt.title("Todos los candidatos")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(resultado_final_rgb)
plt.title(f"Seleccion final: {len(seleccionados)}")
plt.axis("off")

plt.tight_layout()
plt.show()

# ==========================================================
# 11. MOSTRAR ROIS
# ==========================================================

mostrar_rois(resultados)