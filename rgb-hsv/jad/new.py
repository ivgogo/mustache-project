import matplotlib.pyplot as plt
import numpy as np
import cv2

data_file_dir = "/media/ivan/Ivan/jad_spectral/src_Specim-FX17e-076900055547_00.tiff"

example_image = []

_, example_image = cv2.imreadmulti(
    filename=data_file_dir,
    mats=example_image,
    flags=cv2.IMREAD_UNCHANGED
)

example_image = np.asarray(example_image)
example_image = np.transpose(example_image, (0, 2, 1))
#print(example_image.shape)

#index_61 = example_image[:, :, 61]
#index_57 = example_image[:, :, 57]

#result = index_61 - index_57

# conveyor indexes
index_61 = example_image[:, :, 30]
index_57 = example_image[:, :, 25]

result = index_61 - index_57

new_numpy = np.where(result > 0, 1, 0)

plt.imshow(new_numpy)
plt.colorbar() 
plt.show()

'''
dimension = 3  
index1 = 61     
index2 = 57      

# Restar los valores
result = example_image.copy()  # Para no modificar la matriz original
result[:, dimension, index1] -= example_image[:, dimension, index2]

plt.imshow(result)
plt.colorbar() 
plt.show()'''
