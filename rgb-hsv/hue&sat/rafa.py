import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
import cv2

# images_1
data_file_dir = "/media/ivan/Ivan/jad/images_1/plastic/src_Specim-FX17e-076900055547_00.tiff"
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

# =========================================================================================================

y, x = 512, 827     # POINT 512, 827 of image 43 (false positive) - images_3 folder
y2, x2 = 193, 572     # POINT 193, 572 of image 01 (very small plastic) - images_4 folder
# y2, x2 = 108, 303     # POINT 108, 303 of image 04 (very small plastic) - images_4 folder

# points = [25, 80]

# valuesplastic = [0.6172515, 0.45261446]

# signal = example_image[x, y, points]

# value154 = example_image_2[x2, y2, 154]

# # y1 = 3.9372995
# # y2 = 3.0223446

# y1 = 4.737672888
# y2 = 3.474012223

# signal1 = example_image[x, y, :]
# num_points = len(points)
# op = (1/2)*(pow((value154*y1-signal[0]), 2) + pow(value154*y2-signal[1], 2))
# print(op)

#  ==========================================================================================

# mask = np.ones_like(example_image[:, :, 1])
# value154 = example_image_2[x2, y2, 154]
# points = [25, 80]
# y1 = 4.737672888
# y2 = 3.474012223

# for i in range(1000):
#     for j in range(640):
#         signal = example_image[i, j, points]
#         op = (1/2)*(pow((value154*y1-signal[0]), 2) + pow(value154*y2-signal[1], 2))
#         if op < 0.01: # 6 zeros
#             mask[i, j] = 0

# plt.imshow(mask, cmap='gray')
# plt.show()