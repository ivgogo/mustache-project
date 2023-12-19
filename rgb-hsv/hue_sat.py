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
red = example_image[:, :, 53]   # 53
green = example_image[:, :, 57] # 57
blue = example_image[:, :, 61]  # 61

# mustache
extra1 = example_image[:, :, 146]
extra2 = example_image[:, :, 154]
extra3 = example_image[:, :, 159]

# insignia conveyor belt
extra4 = example_image[:, :, 25]
extra5 = example_image[:, :, 30]

#
extra6 = example_image[:, :, 80]


# ====================================== jad ======================================
# point1 = example_image[:, :, 25]
# point2 = example_image[:, :, 44]
# point3 = example_image[:, :, 80]
# point4 = example_image[:, :, 154]
pointd = (example_image[:, :, 61] - example_image[:, :, 57])/4


# ====================================== all points ======================================
point1 = example_image[:, :, 25]
point2 = example_image[:, :, 30]
point3 = example_image[:, :, 39]
point4 = example_image[:, :, 44]
point5 = example_image[:, :, 53]
point6 = example_image[:, :, 57]
point7 = example_image[:, :, 61]
point8 = example_image[:, :, 80]
point9 = example_image[:, :, 112]
point10 = example_image[:, :, 146]
point11 = example_image[:, :, 154]
point12 = example_image[:, :, 159]

points_list = [25,30,39,44,53,57,61,80,112,146,154,159]

# combine to get rgb
# image_rgb = np.stack([red, green, blue], axis=-1)
# plt.imshow(image_rgb)
# plt.show()
# np.save('test.npy', image_rgb)
# sys.exit()

# saturation
# saturation = (3 * np.minimum.reduce([red, blue, green])) / (red + blue + green)

# jad
saturation = (3 * np.minimum.reduce([point1, point2, point3, point4, pointd])) / (point1 + point2 + point3 + point4 + pointd)

# all points
# saturation = (3 * np.minimum.reduce([point1, point2, point3, point4, point5, point6, point7, point8, point9, point10, point11, point12])) / (sum(points_list))

# all points + derivative of region 1 of interest plastic
# saturation = (3 * np.minimum.reduce([point1, point2, point3, point4, point5, point6, point7, point8, point9, point10, point11, point12, pointd])) / (sum(points_list) + pointd)

# hue
hue = np.arccos((1/2*((red-green)+(red-blue)))/np.sqrt(pow((red-green),2)+(red-blue)*(red-blue)))

# compare element by element
condition = np.less_equal(blue, green)

# apply condition
h = np.where(condition, hue, 360 - hue)


# ====================================== plotting ======================================

fig, axs = plt.subplots(1, 2)

# sat
axs[0].imshow(saturation)
axs[0].set_title('Saturation')

# hue
axs[1].imshow(h)
axs[1].set_title('Hue')

plt.tight_layout()
plt.show()