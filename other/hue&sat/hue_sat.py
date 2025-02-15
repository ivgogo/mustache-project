# https://en.wikipedia.org/wiki/RGB_color_model

import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
import math
import cv2
import sys
from skimage import exposure
from PIL import Image
import imageio

# data_file_dir = "/media/ivan/Ivan/jad_spectral/src_Specim-FX17e-076900055547_00.tiff"

# images_1
# data_file_dir = "/media/ivan/Ivan/jad/images_1/plastic/src_Specim-FX17e-076900055547_00.tiff"
# data_file_dir = "/media/ivan/Ivan/jad/images_1/plastic/src_Specim-FX17e-076900055547_01.tiff"
# data_file_dir = "/media/ivan/Ivan/jad/images_1/plastic/src_Specim-FX17e-076900055547_02.tiff"
# data_file_dir = "/media/ivan/Ivan/jad/images_1/plastic/src_Specim-FX17e-076900055547_03.tiff"

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

# images_4
# data_file_dir = "/media/ivan/Ivan/jad/images_4/plastic/src_Specim-FX17e-076900055547_00.tiff"
data_file_dir = "/media/ivan/Ivan/jad/images_4/plastic/src_Specim-FX17e-076900055547_01.tiff"     # Plastic yes
# data_file_dir = "/media/ivan/Ivan/jad/images_4/plastic/src_Specim-FX17e-076900055547_02.tiff"
# data_file_dir = "/media/ivan/Ivan/jad/images_4/plastic/src_Specim-FX17e-076900055547_03.tiff"
# data_file_dir = "/media/ivan/Ivan/jad/images_4/plastic/src_Specim-FX17e-076900055547_04.tiff"       # plastic YES
# data_file_dir = "/media/ivan/Ivan/jad/images_4/plastic/src_Specim-FX17e-076900055547_05.tiff"

example_image = []

_, example_image = cv2.imreadmulti(
    filename=data_file_dir,
    mats=example_image,
    flags=cv2.IMREAD_UNCHANGED
)

example_image = np.asarray(example_image)
example_image = np.transpose(example_image, (0, 2, 1))
print(example_image.shape)

normalized_image = example_image.copy()
example_image = (normalized_image-normalized_image.min())/(normalized_image.max()-normalized_image.min()) * 255

# ====================================== Points of interest for Hue ======================================

# valle característico plástico 
red = example_image[:, :, 58]   # 58
green = example_image[:, :, 53] # 57
blue = example_image[:, :, 61]  # 61

# mejores valores hue --> red=58, green=57 y blue=61 (plásticos grandes)
# mejores valores hue para plásticos pequeños --> red=58, green=53 y blue=61
# red=53, green=58, blue=61 # ok con los 4 mas o menos
# red=53, green=57, blue=61 # ok con los 4 mas o menos
# red=57, green=53, blue=61 # resultados muy parecidos a 58,53,61 solo buenos con los peques

# Mustache
extra1 = example_image[:, :, 146]
extra2 = example_image[:, :, 154]
extra3 = example_image[:, :, 159]
# Conveyor belt points of interest
extra4 = example_image[:, :, 25]
extra5 = example_image[:, :, 30]
# 
extra6 = example_image[:, :, 80]


# ====================================== Points Jad ======================================
point1 = example_image[:, :, 25]
point2 = example_image[:, :, 44]
point3 = example_image[:, :, 80]
point4 = example_image[:, :, 154]
pointd = ((example_image[:, :, 61] - example_image[:, :, 57])/4)*10


# ====================================== All Points ======================================
# point1 = example_image[:, :, 25]
# point2 = example_image[:, :, 30]
# point3 = example_image[:, :, 39]
# point4 = example_image[:, :, 44]
# point5 = example_image[:, :, 53]
# point6 = example_image[:, :, 57]
# point7 = example_image[:, :, 61]
# point8 = example_image[:, :, 80]
# point9 = example_image[:, :, 112]
# point10 = example_image[:, :, 146]
# point11 = example_image[:, :, 154]
# point12 = example_image[:, :, 159]

# points_list = [25,30,39,44,53,57,61,80,112,146,154,159]

# ====================================== Saturation Equations ======================================

# saturation
# saturation = (3 * np.minimum.reduce([red, blue, green])) / (red + blue + green) 
# pretty decent with red = 58, green = 53 and blue = 61
# pretty decent with red = 58, green = 57 and blue = 61 --> la rodea bastante bien  

# jad
saturation = (3 * np.minimum.reduce([point1, point2, point3, point4, pointd])) / (point1 + point2 + point3 + point4 + pointd)   # decent

# all points
# saturation = (3 * np.minimum.reduce([point1, point2, point3, point4, point5, point6, point7, point8, point9, point10, point11, point12])) / (sum(points_list))    # BAD detection

# all points + derivative of region 1 of interest plastic
# saturation = (3 * np.minimum.reduce([point1, point2, point3, point4, point5, point6, point7, point8, point9, point10, point11, point12, pointd])) / (sum(points_list) + pointd)   # not really good

# ====================================== Hue Equations ======================================

# mejores valores hue --> red=58, green=57 y blue=61 (plásticos grandes)
# mejores valores hue para plásticos pequeños --> red=58, green=53 y blue=61
# red=53, green=58, blue=61
# red=53, green=57, blue=61
# red=57, green=53, blue=61

# hue
hue = np.arccos((1/2*((red-green)+(red-blue)))/np.sqrt(pow((red-green),2)+(red-blue)*(red-blue)))

# compare element by element
condition = np.less_equal(blue, green)

# apply condition
h = np.where(condition, hue, 360 - hue)

# Mask
mask = np.where(h>358.895, 0, 1)



m2 = np.where(mask==0)
coords = list(zip(m2[0], m2[1]))
positions = mask[m2]
print(coords)


# ====================================== plotting ======================================

fig, axs = plt.subplots(1,2)

# sat
axs[0].imshow(mask)
axs[0].set_title('Saturation')

# # mask
# axs[0].imshow(mask)
# axs[0].set_title('"Mask"')

# hue
axs[1].imshow(h)
axs[1].set_title('Hue')

plt.tight_layout()
plt.show()