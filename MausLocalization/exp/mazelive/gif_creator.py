#Load all png from a file and create a gif

import imageio
import os
import numpy as np

import imageio
import os
import numpy as np
import os
import imageio

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Construct the base path
base_path = os.path.join(script_dir, 'test_runs', '566')

base_file_name = "latent_epoch_Extra_" 
images = []
for i in range(1, 20):
    try:
        file_name = os.path.join(base_path, base_file_name + str(i) + ".png")
        image = imageio.imread(file_name)
        images.append(image)
    except:
        break

# Construct the output file path
output_file_path = os.path.join(base_path, 'movie.gif')

imageio.mimsave(output_file_path, images, duration=0.1)