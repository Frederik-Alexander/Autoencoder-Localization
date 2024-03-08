"""Datasets."""
import os

import numpy as np

import torch
from torch.utils.data import Dataset
#from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset
import h5py
from sklearn.model_selection import train_test_split
import math 

import matplotlib.pyplot as plt
import os


BASEPATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data'))


class MAZEBASE(Dataset):
    """ CUSTOM MOUSE DATASET"""

    def __init__(self, transform=None, train=True, test_fraction=0.2,
                 seed=42):
        """
        Args:
            root (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.test_fraction=0.2
        self.transform = transform

        data, labels = self._load_data()
        #labels = [math.sqrt(i[0]**2+i[1]**2) for i in labels]
        data_train, data_test, labels_train, labels_test = train_test_split(
            data, labels, test_size=test_fraction,
            random_state=seed)

        if train:
            self.data = data_train
            self.labels = labels_train
        else:
            self.data = data_test
            self.labels = labels_test


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, target


    def _load_data(self):
        import platform
        current_platform = platform.system()

        # Get the current script's directory
        current_dir = os.path.dirname(os.path.realpath(__file__))

        # Define the relative path to the file
        relative_path = "data/3d_maze/3d_maze_dataset_walking_around.pkl"

        # Combine the current directory with the relative path
        filename = os.path.join(current_dir, relative_path)

        import pickle
        def load_maze_data(filename):
            with open(filename, 'rb') as file:
                data = pickle.load(file)
            return data

        # Usage
        data = load_maze_data(filename)
        images = np.array(data['images'])
        positions = np.array(data['positions'])
        directions = np.array(data['directions'])
        ## Concaternating the 2 arrays 
        #pos_dir = np.concatenate((images, positions), axis=0)
        def direction_to_angle(direction):
            dx, dy = direction  # Assuming direction is a tuple (dx, dy)
            angle_radians = math.atan2(dy, dx)
            #angle_degrees = math.degrees(angle_radians)
            return angle_radians
        
        def xy_to_polar (x,y): 

            radius = math.sqrt(x**2 + y**2)
            angle = math.atan2(y, x)  # This returns the angle in radians
            return radius, angle 
        
        xy_in_polar = [ xy_to_polar((positions[i][0]-4.5) /4.5 ,(positions[i][1]-4.5) /4.5 ) for i in range(len(positions))]
        pos_dir = np.array([[xy_in_polar[i][0],xy_in_polar[i][1], direction_to_angle((directions[i][0],directions[i][1]))] for i in range(len(positions))])
        return images, pos_dir





class MAZE(MAZEBASE):

    transforms = transforms.Compose([
        transforms.ToTensor()
        ,transforms.Normalize(
            (0.5,),  # Mean for the single channel
            (0.5,)   # Standard deviation for the single channel
        )
    ])

    def __init__(self, train=True):
        """COIL100 dataset normalized."""
        super().__init__(
             transform=self.transforms, train=train)

    def inverse_normalization(self, normalized):
        """Inverse the normalization applied to the original data.

        Args:
            x: Batch of data

        Returns:
            Tensor with normalization inversed.

        """
        normalized = 0.5 * (normalized + 1)
        return normalized



