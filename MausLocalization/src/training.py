"""Training classes."""
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from .datasets.splitting import split_dataset
import numpy as np
import time 
import math 

class TrainingLoop():
    """Training a model using a dataset."""

    def __init__(self, model, dataset, n_epochs, batch_size, learning_rate,
                 weight_decay=1e-5, device='cuda', callbacks=None, torus_iometrie_loss = False):
        """Training of a model using a dataset and the defined callbacks.

        Args:
            model: AutoencoderModel
            dataset: Dataset
            n_epochs: Number of epochs to train
            batch_size: Batch size
            learning_rate: Learning rate
            callbacks: List of callbacks
        """
        self.model = model
        self.dataset = dataset
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device
        self.callbacks = callbacks if callbacks else []
        self.torus_iometrie_loss = torus_iometrie_loss
    def _execute_callbacks(self, hook, local_variables):
        stop = False
        for callback in self.callbacks:
            # Convert return value to bool --> if callback doesn't return
            # anything we interpret it as False
            stop |= bool(getattr(callback, hook)(**local_variables))
        return stop

    def on_epoch_begin(self, local_variables):
        """Call callbacks before an epoch begins."""
        return self._execute_callbacks('on_epoch_begin', local_variables)

    def on_epoch_end(self, local_variables):
        """Call callbacks after an epoch is finished."""
        return self._execute_callbacks('on_epoch_end', local_variables)

    def on_batch_begin(self, local_variables):
        """Call callbacks before a batch is being processed."""
        self._execute_callbacks('on_batch_begin', local_variables)

    def on_batch_end(self, local_variables):
        """Call callbacks after a batch has be processed."""
        self._execute_callbacks('on_batch_end', local_variables)



    def pos_dir_to_torus_OLD(self, pos_dir): #
        ## Converts an x,y and direction coordiante to a torus coordinate
        R = 0.75 # Distance from the center of the hole to the center of the tube
        r = 0.25 # Radius of the tube
        v = pos_dir[:,0]
        phi = pos_dir[:,1]
        theta = pos_dir[:,2]
 
        x = [(R + v[i] * r * np.cos(phi[i])) * np.cos(theta[i]) for i in range(len(pos_dir))]
        #x = (R + v * r * np.cos(phi)) * np.cos(theta)
        
        y = [ (R + v[i] * r * np.cos(phi[i])) * np.sin(theta[i])for i in range(len(pos_dir))]
        
        #y = (R + v * r * np.cos(phi)) * np.sin(theta)

        z = [v[i] * r * np.sin(phi[i]) for i in range(len(pos_dir))]
        #z = v * r * np.sin(phi)

        
        return x,y,z
    
    def pos_dir_to_torus(self, pos_dir):
        ## Converts an x,y and direction coordiante to a torus coordinate
        R = 1.2 # Distance from the center of the hole to the center of the tube
        r = 0.5 # Radius of the tube
        v = pos_dir[:,0]
        theta = pos_dir[:,1]
        phi = pos_dir[:,2]

        x = (R + v * r * torch.cos(phi)) * torch.cos(theta)
        y = (R + v * r * torch.cos(phi)) * torch.sin(theta)
        z = v * r * torch.sin(phi)

        return x,y,z
        """

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        # Plotting
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Plotting a subset of points for clarity
        ax.scatter(x,y,z, color='b', s=1)

        # Find the maximum and minimum bounds across all dimensions
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0

        # Get the mid points in each dimension
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5

        # Set the limits
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')

        plt.title('Filled Torus')
        plt.show()


        return x,y,z
        """


    def forced_isometrie_loss_OLD(self, labels,latents,epoch=0):
        x_true,y_true,z_true = self.pos_dir_to_torus( labels)
        x,y,z = np.array(latents[:,0].detach().numpy()),np.array(latents[:,1].detach().numpy()),np.array(latents[:,2].detach().numpy())
    
        #norm = np.sqrt((x+latents[:][0])^2,(y+latents[:][0])^2,(z+latents[:][0])^2) 
        #norm calcualted via for loop entry wise
        norm = [np.sqrt((x[i]-x_true[i])**2+ (y[i]-y_true[i])**2+ (z[i]-z_true[i])**2) for i in range(len(x)) ]
        #print(norm  )
        if epoch == 500:
            x,y,z = np.array(latents[:,0].detach().numpy()),np.array(latents[:,1].detach().numpy()),np.array(latents[:,2].detach().numpy())
    
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            # Plotting
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')

            # Plotting a subset of points for clarity
            ax.scatter(x,y,z, color='b', s=1)

            # Find the maximum and minimum bounds across all dimensions
            max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0

            # Get the mid points in each dimension
            mid_x = (x.max()+x.min()) * 0.5
            mid_y = (y.max()+y.min()) * 0.5
            mid_z = (z.max()+z.min()) * 0.5

            # Set the limits
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            ax.set_zlabel('Z axis')

            plt.title('Filled Torus')
            plt.show()


        return norm

    def forced_isometrie_loss(self, labels, latents, epoch=0):
        x_true, y_true, z_true = self.pos_dir_to_torus(labels)
        x, y, z = latents[:, 0], latents[:, 1], latents[:, 2]
        #print("x,y,z of latents", x,y,z)
        #print("true x,y,z", x_true,y_true,z_true)   
        norm = torch.sqrt((x - x_true)**2 + (y - y_true)**2 + (z - z_true)**2)
        if epoch == 70:
            x,y,z = np.array(latents[:,0].detach().numpy()),np.array(latents[:,1].detach().numpy()),np.array(latents[:,2].detach().numpy())
    
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            # Plotting
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')

            # Plotting a subset of points for clarity
            ax.scatter(x,y,z, color='b', s=1)

            # Find the maximum and minimum bounds across all dimensions
            max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0

            # Get the mid points in each dimension
            mid_x = (x.max()+x.min()) * 0.5
            mid_y = (y.max()+y.min()) * 0.5
            mid_z = (z.max()+z.min()) * 0.5

            # Set the limits
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            ax.set_zlabel('Z axis')

            plt.title('Filled Torus')
            plt.show()
        
        return norm



    # pylint: disable=W0641
    def __call__(self):
        """Execute the training loop."""
        model = self.model
        dataset = self.dataset
        n_epochs = self.n_epochs
        batch_size = self.batch_size
        learning_rate = self.learning_rate
        torus_iometrie_loss = self.torus_iometrie_loss
        n_instances = len(dataset)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                  pin_memory=True, drop_last=True)
        n_batches = len(train_loader)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate,
            weight_decay=self.weight_decay)

        epoch = 1
        for epoch in range(1, n_epochs+1):
            if self.on_epoch_begin(remove_self(locals())):
                break

            for batch, (img, label) in enumerate(train_loader):
                ## measuring the time to calcuate one loss step
                start = time.time()
                #print("start: ", start)

                

                if self.device == 'cuda':
                    img = img.cuda(non_blocking=True)

                self.on_batch_begin(remove_self(locals()))

                # Set model into training mode and compute loss
                model.train()
                
                
                loss, loss_components = self.model(img)
                if torus_iometrie_loss==True:
                    isometrie_loss = self.forced_isometrie_loss(label,model.encode(img),epoch=epoch)
                    torus_loss = torch.mean(isometrie_loss)

                
                print ("MSE loss: ", float(loss))
                if torus_iometrie_loss==True:
                    loss =   loss + torus_loss 
                    print("Torus loss: ", float(torus_loss))


                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                # Call callbacks
                self.on_batch_end(remove_self(locals()))
                end = time.time()
                #print("end: ", end) 
                #time in ms
                #print("time: ", (end-start)*1000) # IT takes 1 second 
                


            if self.on_epoch_end(remove_self(locals())):
                break
        return epoch


def remove_self(dictionary):
    """Remove entry with name 'self' from dictionary.

    This is useful when passing a dictionary created with locals() as kwargs.

    Args:
        dictionary: Dictionary containing 'self' key

    Returns:
        dictionary without 'self' key

    """
    del dictionary['self']
    return dictionary

