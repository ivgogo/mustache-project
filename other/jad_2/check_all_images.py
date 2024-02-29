import numpy as np
import hyperspy.api as hs
import matplotlib.pyplot as plt
import cv2
import os
from matplotlib import colors

file_path = '/media/ivan/Ivan/jad'

for subdir in os.listdir(file_path):
    if os.path.isdir(os.path.join(file_path, subdir)):
        for subsubdir in os.listdir(os.path.join(file_path, subdir)):
            subfolder = os.path.join(file_path, subdir, subsubdir)
            if os.path.isdir(subfolder):
                for image in os.listdir(subfolder):
                    if image.endswith(".tiff"):
                        
                        data_file_dir = os.path.join(subfolder, image)
                        # print(data_file_dir)
                        
                        spectral_data = []

                        _, spectral_data = cv2.imreadmulti(
                            filename=data_file_dir,
                            mats=spectral_data,
                            flags=cv2.IMREAD_UNCHANGED
                        )

                        spectral_data = np.asarray(spectral_data)
                        spectral_data = np.transpose(spectral_data, (0, 2, 1))

                        label_correspondences = {
                            "meat": 0,
                            "fat": 1,
                            "conveyor": 2,
                            "plastic": 3
                        }

                        # Add label correspondences, change integer values to change heat map
                        color_correspondences = {
                            label_correspondences["meat"]: "g",
                            label_correspondences["fat"]: "w",
                            label_correspondences["conveyor"]: "b",
                            label_correspondences["plastic"]: "r",
                        }

                        # Initialize segmented image
                        segmented_image = np.zeros(spectral_data.shape[:2], dtype=np.uint8)

                        # Conveyor classification
                        conveyor_pixels = np.where((spectral_data[:, :, 30] - spectral_data[:, :, 25]) > 0.0)
                        segmented_image[conveyor_pixels[0], conveyor_pixels[1]] = 2

                        # Fat classification
                        fat_condition = ((spectral_data[:, :, 80] - spectral_data[:, :, 61])/(80 - 61) - (spectral_data[:, :, 61] - spectral_data[:, :, 53])/(61 - 53))
                        extra_condition = ((spectral_data[:, :, 154] - spectral_data[:, :, 112])/(154 - 112))
                        fat_pixels = np.where(
                            np.logical_and(
                                np.logical_and(
                                    fat_condition > -0.00025,
                                    (spectral_data[:, :, 30] - spectral_data[:, :, 25]) < 0.0
                                ),
                                (spectral_data[:, :, 61] - spectral_data[:, :, 57]) > 0.00045   # 0.0015 before
                            )
                        )

                        segmented_image[fat_pixels[0], fat_pixels[1]] = 1

                        # Plastic classification
                        light_check = (spectral_data[:, :, 61] + spectral_data[:, :, 57]) > 0.05
                        red_check = (spectral_data[:, :, 61] - spectral_data[:, :, 57]) < 0.00045
                        crease_check = (spectral_data[:, :, 153] - spectral_data[:, :, 150]) < 0.0
                        extra_check = (spectral_data[:, :, 60] - spectral_data[:, :, 53]) > 0.001
                        extra_check = False
                        plastic_pixels = np.where(
                            np.logical_and(
                                np.logical_and(red_check, light_check),
                                np.logical_or(crease_check, extra_check)
                            )    
                        )
                        segmented_image[plastic_pixels[0], plastic_pixels[1]] = 3
                        
                        if (3 in segmented_image) and ("normal" in data_file_dir):
                            print(f"False positive in image: {data_file_dir}")
                            # Visualization
                            color_map = colors.ListedColormap([
                                color_correspondences[0],
                                color_correspondences[1],
                                color_correspondences[2],
                                color_correspondences[3]
                            ])
                            print([
                                color_correspondences[0],
                                color_correspondences[1],
                                color_correspondences[2],
                                color_correspondences[3]
                            ])

                            plt.imshow(segmented_image, interpolation="nearest", cmap=color_map, vmin=0, vmax=4)
                            plt.show()
                        else:
                            print(f"{data_file_dir} correctly labeled!")        
                        
                        # With 0.0015 only 1 pixel incorrectly labeled in image 43
                        # With 0.00045 all "correctly labeled"