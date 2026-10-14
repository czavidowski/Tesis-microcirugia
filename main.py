import cv2
import numpy as np
import matplotlib.pyplot as plt
import funciones

px_por_mm, mm_por_px = funciones.calibrar_imagen("regla.jpeg", 20)

# --------------------------------------------------
# 1. CARGAR IMAGEN
# --------------------------------------------------
imagen = cv2.imread("imagen2.jpeg")

if imagen is None:
    raise FileNotFoundError("No se pudo cargar la imagen")

imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)


# --------------------------------------------------
# 2. SEGMENTAR EL HILO AZUL
# --------------------------------------------------
hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

azul_bajo = np.array([90, 60, 40])
azul_alto = np.array([140, 255, 255])

mascara = cv2.inRange(hsv, azul_bajo, azul_alto)

# limpieza suave
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)


# --------------------------------------------------
# 3. TOMAR TODOS LOS PIXELES BLANCOS DE LA MASCARA
# --------------------------------------------------
y, x = np.where(mascara > 0)
puntos = np.column_stack((x, y)).astype(np.float32)

print("Cantidad de píxeles detectados:", len(puntos))


# --------------------------------------------------
# 4. HOTELLING / PCA
# --------------------------------------------------
media = np.mean(puntos, axis=0)
puntos_centrados = puntos - media

cov = np.cov(puntos_centrados.T)

autovalores, autovectores = np.linalg.eigh(cov)

indice_mayor = np.argmax(autovalores)
direccion_principal = autovectores[:, indice_mayor]

vx = direccion_principal[0]
vy = direccion_principal[1]

angulo = np.degrees(np.arctan2(vy, vx))

print("Ángulo del eje principal:", angulo)


# --------------------------------------------------
# 5. ROTAR PARA QUE EL EJE PRINCIPAL QUEDE VERTICAL
# --------------------------------------------------
angulo_rotacion = 90 - angulo
print("Ángulo de rotación:", angulo_rotacion)

imagen_rotada, M = funciones.rotar_sin_recortar(imagen, angulo_rotacion, (255, 255, 255))
mascara_rotada, _ = funciones.rotar_sin_recortar(mascara, angulo_rotacion, 0)

imagen_rotada_rgb = cv2.cvtColor(imagen_rotada, cv2.COLOR_BGR2RGB)


# --------------------------------------------------
# 6. MOSTRAR RESULTADOS
# --------------------------------------------------
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(imagen_rgb)
plt.title("Imagen original")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(mascara, cmap="gray")
plt.title("Máscara azul")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(imagen_rotada_rgb)
plt.title("Imagen rotada a vertical")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(mascara_rotada, cmap="gray")
plt.title("Máscara rotada")
plt.axis("off")

plt.tight_layout()
plt.show()