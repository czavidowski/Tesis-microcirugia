
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# ==========================================================
# 1. SEGMENTAR EL HILO AZUL
# ==========================================================

def segmentar_hilo(imagen):

    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    azul_bajo = np.array([90, 60, 40])
    azul_alto = np.array([140, 255, 255])

    mascara = cv2.inRange(
        hsv, azul_bajo, azul_alto
    )

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
# 2. HOTELLING / PCA
# ==========================================================

def calcular_angulo_pca(mascara):

    y, x = np.where(mascara > 0)

    if len(x) < 2:
        raise ValueError(
            "No hay suficientes pixeles azules para PCA."
        )

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
# 3. ROTAR IMAGEN Y MASCARA
# ==========================================================

def rotar_mismo_tamanio(imagen, angulo, color_fondo,
                       interpolacion=cv2.INTER_LINEAR):

    alto, ancho = imagen.shape[:2]

    centro = (ancho / 2, alto / 2)

    M = cv2.getRotationMatrix2D(
        centro, angulo, 1
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
# 4. DETECTAR CANDIDATOS POR GROSOR
# ==========================================================

def detectar_candidatos(mascara):

    binaria = (mascara > 0).astype(np.uint8)

    distancia = cv2.distanceTransform(
        binaria, cv2.DIST_L2, 5
    )

    # Grosor local máximo en cada fila.
    # En realidad corresponde al radio local estimado.
    perfil = np.max(distancia, axis=1)

    perfil_suave = cv2.GaussianBlur(
        perfil.reshape(-1, 1),
        (1, 0),
        sigmaX=0,
        sigmaY=4
    ).ravel()

    valores = perfil_suave[perfil_suave > 0]

    if len(valores) == 0:
        return [], distancia, perfil_suave, []

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

    picos, propiedades = find_peaks(
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
        y2 = min(mascara.shape[0], y_pico + margen + 1)

        region = distancia[y1:y2, :]

        y_local, x_centro = np.unravel_index(
            np.argmax(region),
            region.shape
        )

        centros.append((
            int(x_centro),
            int(y1 + y_local)
        ))

    return centros, distancia, perfil_suave, picos


# ==========================================================
# 5. ESQUELETIZACION
# ==========================================================

def obtener_esqueleto(mascara_local):

    imagen = (
        (mascara_local > 0).astype(np.uint8) * 255
    )

    esqueleto = np.zeros_like(imagen)

    elemento = cv2.getStructuringElement(
        cv2.MORPH_CROSS, (3, 3)
    )

    while cv2.countNonZero(imagen) > 0:

        erosion = cv2.erode(imagen, elemento)

        apertura = cv2.dilate(erosion, elemento)

        diferencia = cv2.subtract(
            imagen, apertura
        )

        esqueleto = cv2.bitwise_or(
            esqueleto, diferencia
        )

        imagen = erosion

    return esqueleto


# ==========================================================
# 6. ANALIZAR RAMIFICACIONES
# ==========================================================

def analizar_candidatos(mascara, centros):

    alto, ancho = mascara.shape

    resultados = []

    fig, axes = plt.subplots(
        2, len(centros),
        figsize=(3 * len(centros), 7),
        squeeze=False
    )

    for i, (x, y) in enumerate(centros):

        margen = 45

        x1 = max(0, x - margen)
        x2 = min(ancho, x + margen + 1)

        y1 = max(0, y - margen)
        y2 = min(alto, y + margen + 1)

        roi = mascara[y1:y2, x1:x2]

        esqueleto = obtener_esqueleto(roi)

        binario = (esqueleto > 0).astype(np.uint8)

        # Contar los 8 vecinos de cada pixel.
        kernel_vecinos = np.ones(
            (3, 3), dtype=np.float32
        )

        kernel_vecinos[1, 1] = 0

        vecinos = cv2.filter2D(
            binario.astype(np.float32),
            -1,
            kernel_vecinos,
            borderType=cv2.BORDER_CONSTANT
        )

        ramificaciones = (
            (binario > 0) &
            (vecinos >= 3)
        )

        cantidad = int(
            np.count_nonzero(ramificaciones)
        )

        resultados.append(cantidad)

        print(
            f"Candidato {i + 1}: "
            f"{cantidad} pixeles de ramificacion"
        )

        axes[0, i].imshow(
            roi, cmap="gray"
        )

        axes[0, i].set_title(
            f"Candidato {i + 1}"
        )

        axes[1, i].imshow(
            esqueleto, cmap="gray"
        )

        yy, xx = np.where(ramificaciones)

        axes[1, i].scatter(
            xx, yy, color="red", s=12
        )

        axes[1, i].set_title(
            f"Ramificaciones: {cantidad}"
        )

        axes[0, i].axis("off")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()

    return resultados


# ==========================================================
# 7. MAIN
# ==========================================================

imagen = cv2.imread("imagen3.jpg")

if imagen is None:
    raise FileNotFoundError(
        "No se pudo cargar imagen3.jpg"
    )

imagen_rgb = cv2.cvtColor(
    imagen, cv2.COLOR_BGR2RGB
)


# ----------------------------------------------------------
# SEGMENTACION
# ----------------------------------------------------------

mascara = segmentar_hilo(imagen)


# ----------------------------------------------------------
# HOTELLING
# ----------------------------------------------------------

angulo = calcular_angulo_pca(mascara)

angulo_rotacion = angulo - 90

print(f"Angulo PCA: {angulo:.2f}")
print(f"Angulo de rotacion: {angulo_rotacion:.2f}")


# ----------------------------------------------------------
# ROTACION
# ----------------------------------------------------------

imagen_rotada, M = rotar_mismo_tamanio(
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


# ----------------------------------------------------------
# DETECCION POR GROSOR
# ----------------------------------------------------------

centros, mapa_distancia, perfil, picos = (
    detectar_candidatos(mascara_rotada)
)

print("\nCantidad de candidatos:", len(centros))
print("Centros:", centros)


# ----------------------------------------------------------
# DIBUJAR CANDIDATOS
# ----------------------------------------------------------

resultado = imagen_rotada.copy()

for numero, (x, y) in enumerate(centros, start=1):

    cv2.circle(
        resultado,
        (x, y),
        8,
        (0, 0, 255),
        3
    )

    cv2.putText(
        resultado,
        str(numero),
        (x + 15, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2
    )


imagen_rotada_rgb = cv2.cvtColor(
    imagen_rotada, cv2.COLOR_BGR2RGB
)

resultado_rgb = cv2.cvtColor(
    resultado, cv2.COLOR_BGR2RGB
)


# ==========================================================
# 8. VISUALIZACION GENERAL
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
plt.title("Hotelling / PCA")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(mapa_distancia, cmap="hot")
plt.title("Mapa de grosor")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.plot(perfil, np.arange(len(perfil)))

for i, y_pico in enumerate(picos, start=1):

    plt.axhline(
        y_pico,
        color="red",
        linestyle="--",
        alpha=0.5
    )

    plt.text(
        max(perfil) * 0.7,
        y_pico,
        str(i),
        color="red"
    )

plt.gca().invert_yaxis()
plt.xlabel("Radio local estimado [px]")
plt.ylabel("Posicion vertical [px]")
plt.title("Perfil de grosor")

plt.subplot(2, 3, 6)
plt.imshow(resultado_rgb)
plt.title(f"Candidatos: {len(centros)}")
plt.axis("off")

plt.tight_layout()
plt.show()


# ==========================================================
# 9. ANALISIS DEL ESQUELETO
# ==========================================================

if len(centros) > 0:

    ramificaciones = analizar_candidatos(
        mascara_rotada,
        centros
    )

    print("\nRESUMEN FINAL")

    for i, cantidad in enumerate(
        ramificaciones, start=1
    ):

        print(
            f"Candidato {i}: "
            f"{cantidad} pixeles de ramificacion"
        )

else:

    print("No se detectaron candidatos.")
