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
data_file_dir_2 = "/media/ivan/Ivan/jad/images_4/plastic/src_Specim-FX17e-076900055547_01.tiff"     # Plastic yes
# data_file_dir = "/media/ivan/Ivan/jad/images_4/plastic/src_Specim-FX17e-076900055547_02.tiff"
# data_file_dir = "/media/ivan/Ivan/jad/images_4/plastic/src_Specim-FX17e-076900055547_03.tiff"
# data_file_dir_2 = "/media/ivan/Ivan/jad/images_4/plastic/src_Specim-FX17e-076900055547_04.tiff"       # plastic YES
# data_file_dir = "/media/ivan/Ivan/jad/images_4/plastic/src_Specim-FX17e-076900055547_05.tiff"

example_image = []

_, example_image = cv2.imreadmulti(
    filename=data_file_dir,
    mats=example_image,
    flags=cv2.IMREAD_UNCHANGED
)

example_image = np.asarray(example_image)
example_image = np.transpose(example_image, (0, 2, 1))

example_image_2 = []

_, example_image_2 = cv2.imreadmulti(
    filename=data_file_dir_2,
    mats=example_image_2,
    flags=cv2.IMREAD_UNCHANGED
)

example_image_2 = np.asarray(example_image_2)
example_image_2 = np.transpose(example_image_2, (0, 2, 1))

y, x = 512, 827     # POINT 512, 827 of image 43 (false positive) - images_3 folder
y2, x2 = 193, 572     # POINT 193, 572 of image 01 (very small plastic) - images_4 folder
# y2, x2 = 108, 303     # POINT 108, 303 of image 04 (very small plastic) - images_4 folder

signal1 = example_image[x, y, :]
signal2 = example_image_2[x2, y2, :]


# =============================================================================================

# signal_filtered_der1 = signal1.copy()
# for k in range(1, 184-1):
#     signal_filtered_der1[:,k] = signal1[:,k+1] - signal1[:,k-1]

# # The derivatives values are extremely small compared to the signal values, therefore we scale them by 1000 to be at in the same range as the signal that
# # is in interval [0,1] and crop the values outside interval [-0.5,+0.5] 
# signal_filtered_der1 = signal_filtered_der1 * 1000
# signal_filtered_der1=np.where(signal_filtered_der1 < -0.5, -0.5, signal_filtered_der1)
# signal_filtered_der1=np.where(signal_filtered_der1 > 0.5, 0.5, signal_filtered_der1)
test = np.gradient(np.gradient(signal1))
test2 = np.gradient(np.gradient(signal2))
a1 = np.gradient(signal1)
a12 = np.gradient(signal2)
plt.plot(signal1, label='signal1 (false positive)')
plt.plot(signal2, label='signal2 (plastic)')
plt.plot(a1, label='signal1 deriv.1 (false positive)')
plt.plot(a12, label='signal2 deriv.1 (plastic)')
plt.plot(test, label='signal1 deriv.2 (false positive)')
plt.plot(test2, label='signal2 deriv.2 (plastic)')
plt.legend()
plt.show()
