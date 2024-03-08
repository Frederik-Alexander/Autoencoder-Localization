import gym
import numpy as np
import os 
from copy import deepcopy
os.environ['MUJOCO_GL'] = 'glfw'  

# Initialize the environment
env = gym.make('memory_maze:MemoryMaze-9x9-ExtraObs-v0') # memory_maze:MemoryMaze-9x9-ExtraObs-v0
env.reset()
#from gui import all
#import recording
#from gui import recording
#from gui import run_gui 


# Storage for images and positions/directions

def set_state(self, state):
    self.env = deepcopy(state)
    obs = np.array(list(self.env.unwrapped.state))
    return obs


images = []
positions = []
directions = []
layout= []

first_run = 0
for _ in range(10000):  # For 10,000 random images
    print(_)
    env.reset()
    num_random_steps = np.random.randint(1, 4)  # Random number of steps for variability

    for __ in range(10): # There are 6 actions to do:
        action = env.action_space.sample()
        #print(env.state)
        #exit()
        observation, _, _, _ = env.step(action)

    # Extract and store data
    images.append(observation['image'])
    positions.append(observation['agent_pos'])
    directions.append(observation['agent_dir'])
    layout.append(observation['maze_layout'])

    if first_run == 0:

        image = observation['image']
        print(positions)
        # If you need to display this image in a Jupyter notebook, you can use matplotlib
        import matplotlib.pyplot as plt
        plt.imshow(image)
        plt.show()
        first_run = 1

import pickle

# Assuming images, positions, and directions are your lists of data
data = {
    'images': images,
    'positions': positions,
    'directions': directions,
    'maze_layout': layout
}

file_name = 'new_maze_dataset_4mar24_10000.pkl'
# Save data to a file
with open(file_name, 'wb') as file:
    pickle.dump(data, file)

# At this point, you have the images, positions, and directions.
# You may want to save this data to disk, perhaps using numpy or another data format.

def load_maze_data(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

# Usage
data = load_maze_data(file_name)
images = data['images']
positions = data['positions']
directions = data['directions']

"""
import gym
import os 
# Set this if you are getting "Unable to load EGL library" error:
os.environ['MUJOCO_GL'] = 'glfw'  

env = gym.make('memory_maze:MemoryMaze-9x9-v0')
env.reset()

# Number of random steps to take
num_random_steps = 10

# Perform random actions
for _ in range(num_random_steps):
    action = env.action_space.sample()  # Randomly sample an action
    observation, _, _, _ = env.step(action)  # Perform the action

# The observation is now the image of the current state
image = observation

# If you need to display this image in a Jupyter notebook, you can use matplotlib
import matplotlib.pyplot as plt
plt.imshow(image)
plt.show()"""