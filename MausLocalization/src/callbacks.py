"""Callbacks for training loop."""
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from src.evaluation.utils import get_space
from src.visualization import visualize_latents

import numpy as np
import platform
# Hush the linter, child callbacks will always have different parameters than
# the overwritten method of the parent class. Further kwargs will mostly be an
# unused parameter due to the way arguments are passed.
# pylint: disable=W0221,W0613


def _load_data_EXTRA():
    # Here we include data of the mouse just turning in a circle, to see if the circle strucutre is visiable in the latents
    import platform
    current_platform = platform.system()
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    #path = /Users/frederikalexander/My Drive/Computer Science/maus/MausLocalisation/src/datasets/data/3d_maze/only_circle_data.pkl
    filename = os.path.join(script_dir, '', 'datasets/data/3d_maze', "only_circle_data.pkl")

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



    return images, positions

class Callback():
    """Callback for training loop."""

    def on_epoch_begin(self, **local_variables):
        """Call before an epoch begins."""

    def on_epoch_end(self, **local_variables):
        """Call after an epoch is finished."""

    def on_batch_begin(self, **local_variables):
        """Call before a batch is being processed."""

    def on_batch_end(self, **local_variables):
        """Call after a batch has be processed."""


class Progressbar(Callback):
    """Callback to show a progressbar of the training progress."""

    def __init__(self, print_loss_components=False):
        """Show a progressbar of the training progress.

        Args:
            print_loss_components: Print all components of the loss in the
                progressbar
        """
        self.print_loss_components = print_loss_components
        self.total_progress = None
        self.epoch_progress = None

    def on_epoch_begin(self, n_epochs, n_instances, **kwargs):
        """Initialize the progressbar."""
        if self.total_progress is None:
            self.total_progress = tqdm(
                position=0, total=n_epochs, unit='epochs')
        self.epoch_progress = tqdm(
            position=1, total=n_instances, unit='instances')

    def _description(self, loss, loss_components):
        description = f'Loss: {loss:3.3f}'
        if self.print_loss_components:
            description += ', '
            description += ', '.join([
                f'{name}: {value:4.2f}'
                for name, value in loss_components.items()
            ])
        return description

    def on_batch_end(self, batch_size, loss, loss_components, **kwargs):
        """Increment progressbar and update description."""
        self.epoch_progress.update(batch_size)
        description = self._description(loss, loss_components)
        self.epoch_progress.set_description(description)

    def on_epoch_end(self, epoch, n_epochs, **kwargs):
        """Increment total training progressbar."""
        self.epoch_progress.close()
        self.epoch_progress = None
        self.total_progress.update(1)
        if epoch == n_epochs:
            self.total_progress.close()


class SaveReconstructedImages(Callback):
    """Callback to save images of the reconstruction."""

    def __init__(self, path):
        """Save images of the reconstruction.

        Args:
            path: Path to store the images to
        """
        self.path = path

    def on_epoch_end(self, model, dataset, img, epoch, **kwargs):
        print("Running : ON_EPOCH_END")
        #exit()

        #print(img)
        #print("image type:")
        #print(type(img))
        #print(img.shape) torch.Size([100, 3, 64, 64])


        """Save reconstruction images."""
        num_channels = img.shape[1] if img.ndim == 4 else img.shape[0] 

        #print(num_channels)
        #exit()

        #img = dataset.inverse_normalization(img)

        import matplotlib.pyplot as plt
        #print(img[0])

        if img[0].shape[0] == 1:
            image_data = img[0][0].cpu()  # Index the first element of the first dimension
        else:
            image_data = img[0].cpu()
        
        if num_channels == 3:
            image_data = np.transpose(image_data.cpu(), (1, 2, 0))
        plt.imshow(image_data)
        plt.axis('off')  # No axes for cleaner display
        #plt.show()
        #exit()
        model.eval()
        latent = model.encode(img)
        #print(latent)
        #exit()
        reconst = model.decode(latent)


        reconst_1 = reconst.detach()
        #print(reconst[0])

        if reconst_1[0].shape[0] == 1:
            reconst_1 = reconst_1[0][0].cpu() # Index the first element of the first dimension
        else:
            reconst_1 = reconst_1[0].cpu()
        if num_channels == 3:
            reconst_1 = np.transpose(reconst_1.cpu(), (1, 2, 0))
        plt.imshow(reconst_1)
        plt.axis('off')  # No axes for cleaner display
        #plt.show()




        import torch 
        # Inverse normalization for both original and reconstructed images
        img = dataset.inverse_normalization(img)
        reconstructed_image = dataset.inverse_normalization(reconst)
        np.set_printoptions(threshold=np.inf)

        #print(img.shape)
        #print(reconstructed_image.shape)
        #exit()
        concatenated_images = torch.cat((img, reconstructed_image), 3)
        concatenated_images_all = concatenated_images.detach() 
        concatenated_images=concatenated_images.detach() 
        if concatenated_images[0].shape[0] == 1:
            concatenated_images = concatenated_images[0][0].cpu()  # Index the first element of the first dimension
        else:
            concatenated_images = concatenated_images[0].cpu()
        #print(concatenated_images)
        if num_channels == 3:
            concatenated_images = np.transpose(concatenated_images.cpu(), (1, 2, 0))
        #concatenated_images*255
        plt.imshow(concatenated_images, cmap='gray')
        plt.axis('off')  # No axes for cleaner display
        #plt.show()

        # Save the concatenated image
        save_image(concatenated_images_all, os.path.join(self.path, f'epoch_{epoch}.png'))
        plt.close()


        #### HERE I WILL PLOT EXTRA DATA POINTS SUCH AS CIRCLE OR LINE: 


        
        data_extra, labels_extra = _load_data_EXTRA()
        print(type(data_extra))
        print(data_extra.shape)

        tensor = torch.from_numpy(data_extra)

        # Change the order of the dimensions
        tensor = tensor.permute(0, 3, 1, 2)
        import torch
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
        img = (tensor.float() / 255.0 )
        transformed_tensor = transforms_pipeline(img).contiguous()
        img = transformed_tensor
        try:
            img = transformed_tensor.cuda()
        except:
            pass
        latent = model.encode(img)
        reconst = model.decode(latent)
        reconst_1 = reconst.detach()
        if reconst_1[0].shape[0] == 1:
            reconst_1 = reconst_1[0][0].cpu() # Index the first element of the first dimension
        else:
            reconst_1 = reconst_1[0].cpu()
        if num_channels == 3:
            reconst_1 = np.transpose(reconst_1.cpu(), (1, 2, 0))
        plt.imshow(reconst_1)
        plt.axis('off')  # No axes for cleaner display

        # Inverse normalization for both original and reconstructed images
        img = dataset.inverse_normalization(img)
        reconstructed_image = dataset.inverse_normalization(reconst)
        np.set_printoptions(threshold=np.inf)
        concatenated_images = torch.cat((img, reconstructed_image), 3)
        concatenated_images_all = concatenated_images.detach() 
        concatenated_images=concatenated_images.detach() 
        if concatenated_images[0].shape[0] == 1:
            concatenated_images = concatenated_images[0][0].cpu()  # Index the first element of the first dimension
        else:
            concatenated_images = concatenated_images[0].cpu()
        #print(concatenated_images)
        if num_channels == 3:
            concatenated_images = np.transpose(concatenated_images.cpu(), (1, 2, 0))
        #concatenated_images*255
        plt.imshow(concatenated_images, cmap='gray')
        plt.axis('off')  # No axes for cleaner display
        #plt.show()


        ### Saving latens of extra:


        visualize_latents(
            latent.detach(),
            labels_extra,
            save_file=os.path.join(self.path, f'latent_epoch_Extra_{epoch}.png')
        )


class SaveLatentRepresentation(Callback):
    """Callback to save images of the reconstruction."""

    def __init__(self, dataset, path, batch_size=64, device='cuda'):
        """Save images of the reconstruction.

        Args:
            path: Path to store the images to
        """
        self.path = path
        self.dataset = dataset
        self.device = device
        self.data_loader = DataLoader(self.dataset, batch_size=batch_size,
                                      drop_last=True, pin_memory=True)

    def on_epoch_end(self, model, dataset, img, epoch, **kwargs):
        """Save reconstruction images."""
        model.eval()
        latents, labels = get_space(model, self.data_loader, mode='latent',
                                    device=self.device)

        visualize_latents(
            latents,
            labels,
            save_file=os.path.join(self.path, f'latent_epoch_{epoch}.png')
        )
