"""Module to train a model with a dataset configuration."""

from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG = False
#sacred.Experiment(..., save_git_info=False)

import os

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
import torch
import numpy as np
import pandas as pd
import random


import matplotlib
matplotlib.use('TkAgg')  # Use the TkAgg backend
import matplotlib.pyplot as plt

from src.callbacks import Callback, SaveReconstructedImages, \
    SaveLatentRepresentation, Progressbar
from src.datasets.splitting import split_validation
#from src.evaluation.eval import Multi_Evaluation
#from src.evaluation.utils import get_space
from src.training import TrainingLoop
from src.visualization import plot_losses, visualize_latents

from .callbacks import LogDatasetLoss, LogTrainingLoss
from .ingredients import model as model_config
from .ingredients import dataset as dataset_config

from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG = False

EXP = Experiment(
    'training',
    ingredients=[model_config.ingredient, dataset_config.ingredient],
    save_git_info=False
)
EXP.captured_out_filter = apply_backspaces_and_linefeeds


@EXP.config
def cfg():
    n_epochs = 10
    batch_size = 64
    learning_rate = 1e-3
    weight_decay = 1e-5
    val_size = 0.15
    early_stopping = 10
    device = 'cuda'
    quiet = False
    evaluation = {
    
    
    
        'active': False,
        'k_min': 10,
        'k_max': 200,
        'k_step': 10,
        'evaluate_on': 'test',
        'online_visualization': False,
        'save_latents': True,
        'save_training_latents': False
    }
    




class NewlineCallback(Callback):
    """Add newline between epochs for better readability."""
    def on_epoch_end(self, **kwargs):
        print()

def plot_image(data):
    plt.imshow(data)
    plt.axis('off')  # Turn off axis numbers
    plt.show()

def init_live_plot():
    plt.ion()
    fig, ax = plt.subplots()
    sc = ax.scatter([], [])
    ax.set_xlim(-2, 2)  # Adjust these limits based on your expected data range
    ax.set_ylim(-2, 2)
    return ax, sc 

def update_plot(ax, sc ,new_points):
    x, y = zip(*new_points)
    sc.set_offsets(list(zip(x, y)))
    ax.relim()  # Recalculate limits
    ax.autoscale_view()  # Auto-adjust to new data
    plt.draw()
    plt.pause(0.01)






@EXP.command
def live_demo(model_number,device = "cpu"):
    """Sacred wrapped function to run training of model."""
    # LOAD MODEL AND WAIT FOR DATA
    import os


    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Move one directory up
    parent_dir = os.path.dirname(script_dir)

    # Construct the full file path
    path_to_model = os.path.join(parent_dir, 'test_runs', str(model_number), 'model_state.pth')
    print(path_to_model)
    
    #model_path = os.path.join(rundir, 'model_state.pth')
    model_path = path_to_model
 

    if os.path.exists(model_path):
        model = model_config.get_instance()
        model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
        model.to(device)
        print("Loaded pre-trained model")
    else:
        print("model not found")
        exit()
    ##### Live walking aroung and plotting the latent space

    import numpy as np
    import time


    #try:
    new_points = []
    new_points_2 = []
    new_points_3 = []
    ax, sc =init_live_plot()
    ax2, sc2 =init_live_plot()
    ax3, sc3 =init_live_plot()
    counter = 0

    while True:
        env_value_old = ""
        #time.sleep(0.03)
        filename = f"/Users/frederikalexander/My Drive/Computer Science/maus/data_for_live_demo/data_log_{counter}.npy"
        if os.path.exists(filename):
            # Load the numpy array from the file
            env_value = np.load(filename)


            #data_extra, labels_extra = _load_data_EXTRA()
            #print(type(data_extra))
            #print(data_extra.shape)

            #tensor = torch.from_numpy(data_extra)

            #plot_image(env_value)
            # Add your processing logic here
            counter += 1

            tensor = torch.from_numpy(np.array([env_value,env_value]))

            # Change the order of the dimensions
            tensor = tensor.permute(0, 3, 1, 2)

            from torchvision import transforms

            # Define your transformation pipeline
            transforms_pipeline = transforms.Compose([


                transforms.Normalize((0.5,), (0.5,))
            ])

            # Manually create a tensor or use an existing one
            # For example, a random tensor representing an image
            # Let's say it's a 1 channel image of size 28x28
        

            # Apply the transformations
            
            # Normalize the tensor
            # Assuming the original values are in the range [0, 255]
            img = tensor.float()
            img = (tensor.float() / 255.0 )
            #print("sum",np.sum(img.detach().numpy()))

            transformed_tensor = img.contiguous()
            transformed_tensor = transforms_pipeline(img).contiguous()

            
            img = transformed_tensor
            try:
                img = transformed_tensor.cuda()
            except:
                
                pass

            #print(img.shape)
            model.eval()
            latent = model.encode(img)

            #reconst = model.decode(latent)


            print("latent", latent)
            new_x = latent.detach().numpy()[0][0]
            new_y = latent.detach().numpy()[0][1]
            new_points.append((new_x, new_y))
            new_points = new_points[-10:]
            update_plot(ax, sc,new_points )

            new_x = latent.detach().numpy()[0][1]
            new_y = latent.detach().numpy()[0][2]
            new_points_2.append((new_x, new_y))
            new_points_2 = new_points_2[-10:]
            update_plot(ax2, sc2,new_points_2)
            print("updated plot")


            new_x = latent.detach().numpy()[0][0]
            new_y = latent.detach().numpy()[0][2]
            new_points_3.append((new_x, new_y))
            new_points_3 = new_points_3[-10:]
            update_plot(ax3, sc3,new_points_3)
            print("updated plot")
            







        else:
                print("No ne image")
                time.sleep(0.1)  # Wait for new data
    
    
    #except KeyboardInterrupt:
    #    print("ERROR!")
    #    pass

@EXP.command
def latent_sliders(model_number,device = "cpu"):
    """Sacred wrapped function to run training of model."""
    import os


    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Move one directory up
    parent_dir = os.path.dirname(script_dir)

    # Construct the full file path
    path_to_model = os.path.join(parent_dir, 'test_runs', str(model_number), 'model_state.pth')
    print(path_to_model)
    
    #model_path = os.path.join(rundir, 'model_state.pth')
    model_path = path_to_model
 

    if os.path.exists(model_path):
        model = model_config.get_instance()
        model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
        model.to(device)
        print("Loaded pre-trained model")
    else:
        print("model not found")
        exit()


    import tkinter as tk
    from PIL import Image, ImageTk
    import numpy as np

    # Assuming your model is named `model` and it has a `decode` method
    # Also assuming your latent space has 3 dimensions

    def denormalize(tensor):
        return  0.5 * (tensor + 1)

    def plot_image_from_latent_space(*args):
        latent_1 = latent_1_slider.get()
        latent_2 = latent_2_slider.get()
        latent_3 = latent_3_slider.get()
        
        latent_vector = np.array([latent_1, latent_2, latent_3])
        latent_vector = np.array([[latent_1, latent_2, latent_3]])  # Make it 2D
        latent_vector = torch.tensor(latent_vector).float()  # Convert to PyTorch tensor
        
        #decoded_image = model.decode(latent_vector)
        
        #tensor([[ 0.1501, -0.5110,  0.0460],
        #[ 0.1501, -0.5110,  0.0460]], grad_fn=<AddmmBackward0>)
        #latent type <class 'torch.Tensor'>

        #Generate random image
        model.eval()
        #latent_vector_3d = torch.tensor(latent_vector).float()
        decoded_image = model.decode(latent_vector)
        decoded_image = denormalize(decoded_image).detach().numpy()

        # Keep all color channels
        decoded_image = decoded_image[0]
        #print(decoded_image_3d)
        
        # Convert the image to a PIL Image and then to a Tkinter PhotoImage
        # Make sure to convert the image data to 8-bit unsigned integers
        
        # Use make_grid to ensure the image tensor is in the correct format
        from torchvision.utils import save_image, make_grid
        from PIL import Image
        decoded_image = torch.from_numpy(decoded_image)
        grid = make_grid(decoded_image)

        # Convert the tensor to a numpy array
        numpy_image = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

        # Convert the numpy array to a PIL Image
        pil_image = Image.fromarray(numpy_image)
        
        
        #pil_image = Image.fromarray(np.uint8(decoded_image_3d * 255), 'RGB')
        pil_image = pil_image.resize((250, 250))  # Change the size as needed

        tk_image = ImageTk.PhotoImage(pil_image)

        # Update the image_label with the new PhotoImage
        image_label.config(image=tk_image)
        image_label.image = tk_image

    root = tk.Tk()

    latent_1_slider = tk.Scale(root, from_=-1, to=1, resolution=0.01, command=plot_image_from_latent_space)
    latent_1_slider.pack()

    latent_2_slider = tk.Scale(root, from_=-1, to=1, resolution=0.01, command=plot_image_from_latent_space)
    latent_2_slider.pack()

    latent_3_slider = tk.Scale(root, from_=-1, to=1, resolution=0.01, command=plot_image_from_latent_space)
    latent_3_slider.pack()

    image_label = tk.Label(root)
    image_label.pack()

    root.mainloop()
    




@EXP.automain
def train(n_epochs, batch_size, learning_rate, weight_decay, val_size,
          early_stopping, device, quiet, evaluation, _run, _log, _seed, _rnd, torus_iometrie_loss, load_model,load_model_number ):
    
    
    """Sacred wrapped function to run training of model."""
    torch.manual_seed(_seed)
    rundir = None
    try:
        rundir = _run.observers[0].dir
    except IndexError:
        pass

    # LOAD MODEL AND WAIT FOR DATA


    # Load pre-trained model if load_model is True
    if load_model:
        import os

        # Model number
        model_number = load_model_number

        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.realpath(__file__))

        # Construct the full file path
        #/Users/frederikalexander/My Drive/Computer Science/maus/MausLocalisation/test_runs/566/model_state.pth
        path_to_model = os.path.join(script_dir, 'test_runs', str(model_number), 'model_state.pth')
        #model_path = os.path.join(rundir, 'model_state.pth')
        model_path = path_to_model
        import os 

        if os.path.exists(model_path):
            model = model_config.get_instance()
            model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
            model.to(device)
            print("Loaded pre-trained model")
        else:
            print("Pre-trained model file not found. Training a new model.")

    # Get data, sacred does some magic here so we need to hush the linter
    # pylint: disable=E1120,E1123
    dataset = dataset_config.get_instance(train=True)
    train_dataset, validation_dataset = split_validation(
        dataset, val_size, _rnd)
    test_dataset = dataset_config.get_instance(train=False)

    # Get model, sacred does some magic here so we need to hush the linter
    # pylint: disable=E1120
    model = model_config.get_instance()
    model.to(device)

    callbacks = [
        LogTrainingLoss(_run, print_progress=quiet),
        LogDatasetLoss('validation', validation_dataset, _run,
                       print_progress=True, batch_size=batch_size,
                       early_stopping=early_stopping, save_path=rundir,
                       device=device),
        LogDatasetLoss('testing', test_dataset, _run, print_progress=True,
                       batch_size=batch_size, device=device),
    ]

    if quiet:
        # Add newlines between epochs
        callbacks.append(NewlineCallback())
    else:
        callbacks.append(Progressbar(print_loss_components=True))

    # If we are logging this run save reconstruction images
    if rundir is not None:
        if hasattr(dataset, 'inverse_normalization'):
            # We have image data so we can visualize reconstructed images
            callbacks.append(SaveReconstructedImages(rundir))
        
        callbacks.append(
            SaveLatentRepresentation(
                train_dataset, rundir, batch_size=64, device=device)
        )

    training_loop = TrainingLoop(
        model, dataset, n_epochs, batch_size, learning_rate, weight_decay,
        device, callbacks,torus_iometrie_loss
    )
    # Run training
    training_loop()

    if rundir:
        import os
        # Save model state (and entire model)
        print('Loading model checkpoint prior to evaluation...')
        state_dict = torch.load(os.path.join(rundir, 'model_state.pth'))
        model.load_state_dict(state_dict)
    model.eval()

    logged_averages = callbacks[0].logged_averages
    logged_stds = callbacks[0].logged_stds
    loss_averages = {
        key: value for key, value in logged_averages.items() if 'loss' in key
    }
    loss_stds = {
        key: value for key, value in logged_stds.items() if 'loss' in key
    }
    metric_averages = {
        key: value for key, value in logged_averages.items() if 'metric' in key
    }
    metric_stds = {
        key: value for key, value in logged_stds.items() if 'metric' in key
    }
    if rundir:
        plot_losses(
            loss_averages,
            loss_stds,
            save_file=os.path.join(rundir, 'loss.png')
        )
        plot_losses(
            metric_averages,
            metric_stds,
            save_file=os.path.join(rundir, 'metrics.png')
        )

    result = {
        key: values[-1] for key, values in logged_averages.items()
    }

    #SAVING THE MODEL: 
    saving =True
    if saving == True:
        print('Loading model checkpoint prior to evaluation...')
        state_dict = torch.load(os.path.join(rundir, 'model_state.pth'))
        model.load_state_dict(state_dict)



    return result

