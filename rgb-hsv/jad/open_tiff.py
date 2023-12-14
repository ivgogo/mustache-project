import rasterio
from rasterio.plot import show

# Ruta de la imagen hiperespectral en formato TIFF
ruta_imagen = "/media/ivan/Ivan/jad_spectral/src_Specim-FX17e-076900055547_00.tiff"

# Abrir la imagen con rasterio
with rasterio.open(ruta_imagen) as src:
    # Mostrar la imagen
    show(src)