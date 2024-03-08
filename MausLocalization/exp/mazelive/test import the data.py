import numpy as np
import matplotlib.pyplot as plt
import os

# Path to the folder containing .npz files
folder_path_images = '/Users/frederikalexander/My Drive/Computer Science/Bachelorarbeit Maus Topo/3D_MAZE/memory-maze-main/train-part0'
#folder_path_eval = '/Users/frederikalexander/My Drive/Computer Science/Bachelorarbeit Maus Topo/3D_MAZE/memory-maze-main/eval'
# List all .npz files in the folder
npz_files = [f for f in os.listdir(folder_path_images) if f.endswith('.npz')]

# Process each file
for file_name in npz_files:
    # Construct the full file path
    file_path = os.path.join(folder_path_images, file_name)
    
    # Load the .npz file
    with np.load(file_path) as data:
        # Assuming the image data is stored under the key 'image'
        image = data['image']
        #image = data['maze_layout']
        print(image.shape)
        # Display the image
        plt.imshow(image[0])
        plt.imshow(image[1])
        plt.imshow(image[2])
        plt.title(file_name)
        plt.show()

