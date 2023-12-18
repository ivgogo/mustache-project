# https://en.wikipedia.org/wiki/RGB_color_model

import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
import cv2
import sys
from skimage import exposure
from PIL import Image
import imageio

data_file_dir = "/media/ivan/Ivan/jad_spectral/src_Specim-FX17e-076900055547_00.tiff"

example_image = []

_, example_image = cv2.imreadmulti(
    filename=data_file_dir,
    mats=example_image,
    flags=cv2.IMREAD_UNCHANGED
)

example_image = np.asarray(example_image)
example_image = np.transpose(example_image, (0, 2, 1))
print(example_image.shape)


# valle característico plástico 
red = example_image[:, :, 53]
green = example_image[:, :, 57]
blue = example_image[:, :, 61]

# mustache
extra1 = example_image[:, :, 146]
extra2 = example_image[:, :, 154]
extra3 = example_image[:, :, 159]

# insignia conveyor belt
extra4 = example_image[:, :, 25]
extra5 = example_image[:, :, 30]

#
extra6 = example_image[:, :, 80]

# combine to get rgb
#imagen_rgb = np.stack([red, green, blue], axis=-1)

# Tu código para obtener la matriz float32 (imagen_rgb)
imagen_rgb = np.stack([red, green, blue], axis=-1)
#imagen_rgb = imagen_rgb.astype(np.uint8)
#im = imagen_rgb.save('test.tiff')
# Guardar la imagen en formato TIFF de punto flotante
#imageio.imsave("output_image.tiff", imagen_rgb)
#print(imagen_rgb.shape)
np.save('test.npy', imagen_rgb)

sys.exit()

# saturation
saturation = (3 * np.minimum.reduce([red, blue, green])) / (red + blue + green)

# Muestra la imagen de saturación
plt.imshow(saturation, cmap='viridis')
plt.title('Saturation')
plt.axis('off')
plt.show()
