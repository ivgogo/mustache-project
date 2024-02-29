import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
import cv2
import sys

# images_3
# data_file_dir = "/media/ivan/Ivan/jad/images_3/normal/src_Specim-FX17e-076900055547_03.tiff"
# data_file_dir = "/media/ivan/Ivan/jad/images_3/normal/src_Specim-FX17e-076900055547_11.tiff"
# data_file_dir = "/media/ivan/Ivan/jad/images_3/normal/src_Specim-FX17e-076900055547_21.tiff"
# data_file_dir = "/media/ivan/Ivan/jad/images_3/normal/src_Specim-FX17e-076900055547_27.tiff"
# data_file_dir = "/media/ivan/Ivan/jad/images_3/normal/src_Specim-FX17e-076900055547_32.tiff"
# data_file_dir = "/media/ivan/Ivan/jad/images_3/normal/src_Specim-FX17e-076900055547_37.tiff"
data_file_dir = "/media/ivan/Ivan/jad/images_3/normal/src_Specim-FX17e-076900055547_43.tiff"
# data_file_dir = "/media/ivan/Ivan/jad/images_3/normal/src_Specim-FX17e-076900055547_51.tiff"
# data_file_dir = "/media/ivan/Ivan/jad/images_3/normal/src_Specim-FX17e-076900055547_54.tiff"
# data_file_dir = "/media/ivan/Ivan/jad/images_3/normal/src_Specim-FX17e-076900055547_69.tiff"
# data_file_dir = "/media/ivan/Ivan/jad/images_3/normal/src_Specim-FX17e-076900055547_71.tiff"

# images_4
# data_file_dir = "/media/ivan/Ivan/jad/images_4/plastic/src_Specim-FX17e-076900055547_00.tiff"
# data_file_dir = "/media/ivan/Ivan/jad/images_4/plastic/src_Specim-FX17e-076900055547_01.tiff"     # Plastic yes
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

# 512, 827 | 43 image3-false positive

# 193, 572 | 01 image4-plastic

# 108, 303 | 04 image4-plastic

# y, x = 512, 827     # POINT 512, 827 of image 43 (false positive) - images_3 folder
y, x = 511, 827
# y, x = 193, 572     # POINT 193, 572 of image 01 (very small plastic) - images_4 folder
# y, x = 108, 303     # POINT 108, 303 of image 04 (very small plastic) - images_4 folder

points = [example_image[x, y, 25], example_image[x, y, 44], example_image[x, y, 80], example_image[x, y, 154]]
mean_value = np.mean(points)

print(f"Mean for point ({x}, {y}) of {data_file_dir}: {mean_value}")