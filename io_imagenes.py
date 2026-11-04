from pathlib import Path


def obtener_imagenes(carpeta="imagenes"):

    carpeta = Path(carpeta)

    if not carpeta.exists():
        raise FileNotFoundError(
            f"No existe la carpeta: {carpeta}"
        )

    extensiones_validas = {
        ".jpg",
        ".jpeg",
        ".png"
    }

    rutas = []

    for archivo in carpeta.rglob("*"):

        if (
            archivo.is_file()
            and archivo.suffix.lower() in extensiones_validas
        ):
            rutas.append(str(archivo))

    rutas.sort()

    return rutas