import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
import cv2
import sys
from skimage import exposure
from PIL import Image
from scipy.ndimage import convolve

# 512, 827 43 image3
# 193, 572 01 image4
# 108, 303 04 image4

# data_file_dir = "/media/ivan/Ivan/jad/images_1/plastic/src_Specim-FX17e-076900055547_00.tiff"
# data_file_dir = "/media/ivan/Ivan/jad/images_1/plastic/src_Specim-FX17e-076900055547_01.tiff"
# data_file_dir = "/media/ivan/Ivan/jad/images_1/plastic/src_Specim-FX17e-076900055547_02.tiff"
# data_file_dir = "/media/ivan/Ivan/jad/images_1/plastic/src_Specim-FX17e-076900055547_03.tiff"

# images_4
# data_file_dir = "/media/ivan/Ivan/jad/images_4/plastic/src_Specim-FX17e-076900055547_00.tiff"
# data_file_dir = "/media/ivan/Ivan/jad/images_4/plastic/src_Specim-FX17e-076900055547_01.tiff"     # Plastic yes
# data_file_dir = "/media/ivan/Ivan/jad/images_4/plastic/src_Specim-FX17e-076900055547_02.tiff"
# data_file_dir = "/media/ivan/Ivan/jad/images_4/plastic/src_Specim-FX17e-076900055547_03.tiff"
data_file_dir = "/media/ivan/Ivan/jad/images_4/plastic/src_Specim-FX17e-076900055547_04.tiff"       # plastic YES
# data_file_dir = "/media/ivan/Ivan/jad/images_4/plastic/src_Specim-FX17e-076900055547_05.tiff"

# images_3
# data_file_dir = "/media/ivan/Ivan/jad/images_3/normal/src_Specim-FX17e-076900055547_03.tiff"
# data_file_dir = "/media/ivan/Ivan/jad/images_3/normal/src_Specim-FX17e-076900055547_11.tiff"
# data_file_dir = "/media/ivan/Ivan/jad/images_3/normal/src_Specim-FX17e-076900055547_21.tiff"
# data_file_dir = "/media/ivan/Ivan/jad/images_3/normal/src_Specim-FX17e-076900055547_27.tiff"
# data_file_dir = "/media/ivan/Ivan/jad/images_3/normal/src_Specim-FX17e-076900055547_32.tiff"
# data_file_dir = "/media/ivan/Ivan/jad/images_3/normal/src_Specim-FX17e-076900055547_37.tiff"
# data_file_dir = "/media/ivan/Ivan/jad/images_3/normal/src_Specim-FX17e-076900055547_43.tiff"
# data_file_dir = "/media/ivan/Ivan/jad/images_3/normal/src_Specim-FX17e-076900055547_51.tiff"
# data_file_dir = "/media/ivan/Ivan/jad/images_3/normal/src_Specim-FX17e-076900055547_54.tiff"
# data_file_dir = "/media/ivan/Ivan/jad/images_3/normal/src_Specim-FX17e-076900055547_69.tiff"
# data_file_dir = "/media/ivan/Ivan/jad/images_3/normal/src_Specim-FX17e-076900055547_71.tiff"

example_image = []

_, example_image = cv2.imreadmulti(
    filename=data_file_dir,
    mats=example_image,
    flags=cv2.IMREAD_UNCHANGED
)

example_image = np.asarray(example_image)
example_image = np.transpose(example_image, (0, 2, 1))
print(example_image.shape)

# ========================================= Points =========================================

point0 = example_image[:, :, 0]
point25 = example_image[:, :, 25]
point39 = example_image[:, :, 39]
point44 = example_image[:, :, 44]
point53 = example_image[:, :, 53]
point57 = example_image[:, :, 57]
point61 = example_image[:, :, 61]
point80 = example_image[:, :, 80]
point112 = example_image[:, :, 112]
point146 = example_image[:, :, 146]
point154 = example_image[:, :, 154]
point159 = example_image[:, :, 159]

points_list = [point0, point25, point39, point44, point53, point57, point61, point80, point112, point146, point154, point159]

# ========================================= Peaks =========================================

# Points 25 ,44, 80, 154

peaks = [point25, point44, point80, point154]
peaks_sum = sum(peaks)/len(peaks)

# ========================================= N1 =========================================

slope1 = np.abs((point0-point25)/(0-25)/10)
slope2 = np.abs((point25-point39)/(25-39)/10)
slope3 = np.abs((point39-point44)/(39-44)/10)
slope4 = np.abs((point44-point53)/(44-53)/10)
slope5 = np.abs((point53-point57)/(53-57)/10)
slope6 = np.abs((point57-point61)/(57-61)*500)
slope7 = np.abs((point61-point80)/(61-80)*10)
slope8 = np.abs((point80-point112)/(80-112)/10)
slope9 = np.abs((point112-point146)/(112-146)/10)
slope10 = np.abs((point146-point159)/(146-159)/10)
slope_extra = np.abs((point53-point61)/(61-53)/100)

all_slopes_1 = [slope1, slope2, slope3, slope4, slope5, slope6, slope7, slope8, slope9, slope10]

# ========================================= N2 =========================================

slope2_1 = np.abs((point25-point44)/(25-44))
slope2_2 = np.abs((point57-point61)/(57-61)*500)
slope2_3 = np.abs((point44-point80)/(44-80))
slope2_4 = np.abs((point80-point112)/(80-112))
slope2_5 = np.abs((point112-point154)/(112-154))
slope_extra = np.abs((point53-point61)/(61-53)/500)

all_slopes_2 = [slope2_1, slope2_2, slope2_3, slope2_4, slope2_5, slope_extra]
all_slopes_2_we = [slope2_1, slope2_2, slope2_3, slope2_4, slope2_5]

# ========================================= Sat Equations (slopes N1 and N2) =========================================

saturation1 = (3 * np.minimum.reduce([slope1, slope2, slope3, slope4, slope5, slope6, slope7, slope8, slope9, slope10])) / (sum(all_slopes_1))
saturation2 = (3 * np.minimum.reduce([slope2_1, slope2_2, slope2_3, slope2_4, slope2_5, slope_extra])) / (sum(all_slopes_2)) 

# normalize
saturation1 = (saturation1-saturation1.min())/(saturation1.max()-saturation1.min()) * 255
saturation2 = (saturation2-saturation2.min())/(saturation2.max()-saturation2.min()) * 255

# ========================================= Sat Equations (only peaks) =========================================

saturation3 = (3 * np.minimum.reduce(peaks)) / (sum(peaks))
saturation3 = (saturation3-saturation3.min())/(saturation3.max()-saturation3.min()) * 255

# ========================================= Sat Equations (slopes N2 + peaks_sum) =========================================

# quitar slope_extra de all_slopes_2
saturation4 = (3 * np.minimum.reduce([slope2_1, slope2_2, slope2_3, slope2_4, slope2_5, peaks_sum])) / (sum(all_slopes_2_we + peaks_sum))
saturation4 = (saturation4-saturation4.min())/(saturation4.max()-saturation4.min()) * 255

# ========================================= Conclusions =========================================

'''
-Best combinations for detection:
Best for small plastic detection --> Slopes 2 + slope_extra
Bigger plastic --> Slopes 2 + peaks_sum
'''

# ====================================== Tests ======================================

# # Define el tamaño del kernel para calcular la suma de los valores de los píxeles vecinos
# tam_kernel = (9, 9)  # Tamaño del kernel para la convolución en cada dimensión

# # Crea el kernel con unos para que la convolución calcule la suma de los valores de los píxeles vecinos
# kernel = np.ones(tam_kernel)

# # Realiza la convolución para calcular la suma de los valores de los píxeles vecinos para cada píxel
# # Asegúrate de especificar el modo de convolución para que la salida tenga el mismo tamaño que la entrada
# suma_pixeles_vecinos = convolve(saturation2, kernel, mode='constant')

# # Supongamos que tenemos un umbral de suma de píxeles vecinos
# umbral = 500

# # Crea una máscara booleana donde True indica que la suma de los píxeles vecinos es mayor que el umbral
# mascara_superacion_umbral = suma_pixeles_vecinos > umbral

# # Utiliza la máscara para identificar los píxeles que superan el umbral
# pixeles_superan_umbral = saturation2.copy()  # Copia de la imagen original
# pixeles_superan_umbral[~mascara_superacion_umbral] = np.nan  # Establece los píxeles que no superan el umbral como NaN

# # Visualiza la máscara de los píxeles que superan el umbral
# plt.imshow(pixeles_superan_umbral, cmap='gray')
# plt.colorbar()
# plt.title('Píxeles que superan el umbral')
# plt.show()

# ====================================== Plotting ======================================

fig, axs = plt.subplots(1, 2)

# sat
axs[0].imshow(saturation4)
axs[0].set_title('Saturation Map 1')

# hue
axs[1].imshow(saturation2)
axs[1].set_title('Saturation Map 2')

plt.tight_layout()
plt.show()