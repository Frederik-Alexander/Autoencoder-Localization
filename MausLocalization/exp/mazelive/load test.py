import gym
import numpy as np
import os 
import pickle
os.environ['MUJOCO_GL'] = 'glfw'  


import pickle
def load_maze_data(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

# Usage
data = load_maze_data("/Users/frederikalexander/My Drive/Computer Science/maus/MausLocalisation/only_circle_data.pkl")
#data = load_maze_data('new_maze_dataset_4mar24_10000.pkl')
images = data['images']
positions = data['positions']
directions = data['directions']
#print(images)

import matplotlib.pyplot as plt
#plt.imshow(images[1])
#plt.show()

# Iterate over the first 10 images
#Print len of data:
#print(len(images))
#exit()
for i in range(100):
    plt.imshow(images[i])
    plt.show()
    #print(positions[i])
    #print(directions[i])
    # The positions are a list with 2 entries, plot them as a point
    plt.scatter(positions[i][0], positions[i][1])
plt.show()