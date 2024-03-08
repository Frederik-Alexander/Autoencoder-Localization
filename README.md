Geo-Locator using an autoencoder

L2 Norm:

Finds pictures in the Dataset that have similar L2-Norms but are far apart

MausLocalization:

Main Program for training of NN and generating maze data

MazeLive:

Customized 3D Maze for data generation and evaluation

Experiments:

Contains the JSON config files for different runs/experiments.

Data:

Contains the data for the runs

Run with the command line:

Latent Sliders:
	python3 -m exp.main_script latent_sliders with model_number="622" device='cpu' experiments/MAZE_TEST/MAZE_coil.json

Train NN

	python3 -m exp.main_script -F test_runs with experiments/MAZE_TEST/MAZE_coil.json device='cpu'

