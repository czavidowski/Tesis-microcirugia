import funciones

px_por_mm, mm_por_px = funciones.calibrar_imagen("escala.jpeg", 20)

print(f"Píxeles por mm: {px_por_mm:.2f}")
print(f"Milímetros por píxel: {mm_por_px:.4f}")