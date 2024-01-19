import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
import cv2
import sys
from skimage import exposure
from PIL import Image

# 512, 827 43 image3
# 193, 572 01 image4
# 108, 303 04 image4

data_file_dir = "/media/ivan/Ivan/jad/images_1/plastic/src_Specim-FX17e-076900055547_00.tiff"
# data_file_dir = "/media/ivan/Ivan/jad/images_1/plastic/src_Specim-FX17e-076900055547_01.tiff"
# data_file_dir = "/media/ivan/Ivan/jad/images_1/plastic/src_Specim-FX17e-076900055547_02.tiff"
# data_file_dir = "/media/ivan/Ivan/jad/images_1/plastic/src_Specim-FX17e-076900055547_03.tiff"

# images_4
# data_file_dir = "/media/ivan/Ivan/jad/images_4/plastic/src_Specim-FX17e-076900055547_00.tiff"
# data_file_dir = "/media/ivan/Ivan/jad/images_4/plastic/src_Specim-FX17e-076900055547_01.tiff"     # Plastic yes
# data_file_dir = "/media/ivan/Ivan/jad/images_4/plastic/src_Specim-FX17e-076900055547_02.tiff"
# data_file_dir = "/media/ivan/Ivan/jad/images_4/plastic/src_Specim-FX17e-076900055547_03.tiff"
# data_file_dir = "/media/ivan/Ivan/jad/images_4/plastic/src_Specim-FX17e-076900055547_04.tiff"       # plastic YES
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

# ========================================= points =========================================

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

# ========================================= peaks =========================================

'''
Points 25 ,44, 80, 154
'''

peaks = [point25, point44, point80, point154]
peaks_sum = sum(peaks)/len(peaks)

# ========================================= N1 =========================================

slope1 = np.abs((point0-point25)/(0-25))
slope2 = np.abs((point25-point39)/(25-39))
slope3 = np.abs((point39-point44)/(39-44))
slope4 = np.abs((point44-point53)/(44-53))
slope5 = np.abs((point53-point57)/(53-57))
slope6 = np.abs((point57-point61)/(57-61)*100)
slope7 = np.abs((point61-point80)/(61-80)/20)
slope8 = np.abs((point80-point112)/(80-112))
slope9 = np.abs((point112-point146)/(112-146))
slope10 = np.abs((point146-point159)/(146-159))

all_slopes_1 = [slope1, slope2, slope3, slope4, slope5, slope6, slope7, slope8, slope9, slope10]

# ========================================= N2 =========================================

slope2_1 = np.abs((point25-point44)/(25-44))
slope2_2 = np.abs((point57-point61)/(57-61)*20000)
slope2_3 = np.abs((point44-point80)/(44-80)/2000)
slope2_4 = np.abs((point80-point112)/(80-112))
slope2_5 = np.abs((point112-point154)/(112-154))

all_slopes_2 = [slope2_1, slope2_2, slope2_3, slope2_4, slope2_5]

saturation1 = (3 * np.minimum.reduce([slope1, slope2, slope3, slope4, slope5, slope6, slope7, slope8, slope9, slope10])) / (sum(all_slopes_1))
saturation2 = (3 * np.minimum.reduce([slope2_1, slope2_2, slope2_3, slope2_4, slope2_5])) / (sum(all_slopes_2))  #  + peaks_sum

# normalize
saturation1 = (saturation1-saturation1.min())/(saturation1.max()-saturation1.min()) * 255
saturation2 = (saturation2-saturation2.min())/(saturation2.max()-saturation2.min()) * 255

# ====================================== plotting ======================================

fig, axs = plt.subplots(1, 2)

# sat
axs[0].imshow(saturation1)
axs[0].set_title('Saturation Map 1')

# hue
axs[1].imshow(saturation2)
axs[1].set_title('Saturation Map 2')

plt.tight_layout()
plt.show()