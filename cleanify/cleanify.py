#!/usr/bin/python3
import numpy as np
import os # OS module in Python 
import matplotlib.pyplot as plt
import glob # Unix style pathname pattern expansion
import cv2 # OpenCV has four functions for blurring
import torch # PyTorch provides tensor computation and deep neural networks 
import torchvision # popular datasets, model architectures, and image transformations for computer vision
import torch.nn as nn # Neural Network layers
import torch.nn.functional as F # NN functions. Used in CleanerCNN class for loss function
import torch.optim as optim  # Optimization algorithms
import time
import argparse # Parser for command-line options, arguments and sub-commands

from tqdm import tqdm # Progress bar
from torch.utils.data import Dataset, DataLoader # Dataset stors samples and their labels, Loader wraps an iterable around the dataset
from torchvision.transforms import transforms # Common image transformations
from torchvision.utils import save_image # Save a given tensor into an image file
from sklearn.model_selection import train_test_split # Split arrays or matrices into random train and test subsets


class CleanDataset(Dataset):
    '''A dataset object that reads images sequentially from one or two
    lists, applying any specified transforms.

    :param input_dir: Path to directory containing input images. The
        input_dir must contain subdirectories `dirty` and `clean`.
    :type input_dir: str
    :param dirty_paths: List of file paths containing dirtified images.
    :type dirty_paths: list
    :param clean_paths: List of file paths containing clean images.
    :type clean_paths: list, optional
    :param transform: Transforms to apply to each image.
    :type transform: torchvision.transforms.transforms object,
        optional.
    '''
    def __init__(self, input_dir, dirty_paths, clean_paths=None,
                 transform=None):
        self._input_dir = input_dir
        self._x = dirty_paths
        self._y = clean_paths
        self._transform = transform
         
    def __len__(self):
        return len(self._x)
    
    def __getitem__(self, i):
        dirty_image = cv2.imread(f"{self._input_dir}/dirty/{self._x[i]}")
        
        if self._transform is not None:
            dirty_image = self._transform(dirty_image)
            
        if self._y is not None:
            clean_image = cv2.imread(f"{self._input_dir}/clean/{self._y[i]}")
            if self._transform is not None:
                clean_image = self._transform(clean_image)
            return (dirty_image, clean_image)
        else:
            return dirty_image


class CleanerConvAE(nn.Module):
    '''A convolutional autoencoder neural network.'''
    def __init__(self):
        '''Constructor method.'''
        super(CleanerConvAE, self).__init__()

        self.enc = [
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3),
        ]

        self.dec = [
            nn.ConvTranspose2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3),
        ]

    def forward(self, x):
        '''Overrides nn.Module.forward.'''
        for enc in self.enc:
            x = F.relu(enc(x))

        for enc in self.dec:
            x = F.relu(dec(x))

        return x


class CleanerCNN(nn.Module):
    '''A convolutional neural network.'''
    def __init__(self):
        '''Constructor method.'''
        super(CleanerCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9,
                               padding=4, bias=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1,
                               padding=0, bias=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5,
                               padding=2)


    def forward(self, x):
        '''Overrides nn.Module.forward.'''
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.conv3(x)

        return x


class Cleaner():
    '''A wrapper for a convolutional neural network (CNN) that cleans
    distorted images. This class represents the entire cleaner object
    including all relevant parameters. The network itself is a
    component object.

    :param epochs: Number of epochs for which to train the cleaner.
    :type epochs: int, optional
    :param use_autoencoder: Whether to use a convolutional autoencoder
        rather than a plain CNN.
    :type use_autoencoder: bool, optional
    '''
    def __init__(self, epochs=40, use_autoencoder=False):
        '''Constructor method.'''
        self._epochs = epochs
        self._use_autoencoder = use_autoencoder

        self._input_dir = 'input'
        self._output_dir = 'output'
        os.makedirs(self._output_dir, exist_ok=True)

        self._device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        dirty_files = os.listdir(f'{self._input_dir}/dirty')
        clean_files = os.listdir(f'{self._input_dir}/clean')
        for file_list in [dirty_files, clean_files]:
            # For macOS: Skip invisible Desktop Services Store file.
            if '.DS_Store' in file_list:
                file_list.remove('.DS_Store')
            file_list.sort()

        # Split into random training and validation subsets
        (x_train, x_val, y_train, y_val) = train_test_split(dirty_files,
                                                            clean_files,
                                                            test_size=0.25)
        print(f'Train data instances: {len(x_train)}')
        print(f'Validation data instances: {len(x_val)}')

        # Use Torchvision Compose() to combine transforms into one call.
        # ToPILImage makes a Pillow image.
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.ToTensor(),
        ])
        train_data = CleanDataset(self._input_dir, x_train, y_train, transform)
        val_data = CleanDataset(self._input_dir, x_val, y_val, transform)

        # Load the training and validation datasets, and shuffle.
        # Each dataset has a batch size of 2: one for original the other
        # for distorted.
        batch_size = 2
        self._trainloader = DataLoader(train_data, batch_size=batch_size,
                                       shuffle=True)
        self._valloader = DataLoader(val_data, batch_size=batch_size,
                                     shuffle=False)

        # to() moves and/or casts the parameters and buffers.
        # device is cuda:0 or cpu.
        # Returns self.
        if self._use_autoencoder:
            self._model = CleanerConvAE().to(self._device)
        else:
            self._model = CleanerCNN().to(self._device)

        # Loss function: Mean Squared Error
        self._criterion = nn.MSELoss()

        # The optimizer holds the current state and updates the
        # parameters.
        # lr is learning rate.
        # Adam is proposed in https://arxiv.org/abs/1412.6980.
        self._optimizer = optim.Adam(self._model.parameters(), lr=0.001)

        # Reduce the learning rate based on how training is going
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self._optimizer,
                mode='min',
                patience=5,
                factor=0.5,
                verbose=True
        )


    # Run through all the data once, returning the training loss for the
    # epoch.
    def _fit(self):
        # Put the model in training mode
        self._model.train()
        running_loss = 0.0
        n_iter = len(self._trainloader.dataset) // self._trainloader.batch_size

        for data in tqdm(self._trainloader, total=n_iter):
            dirty_image = data[0].to(self._device)
            clean_image = data[1].to(self._device)

            # Set the gradients of all optimized torch Tensors to zero
            self._optimizer.zero_grad()

            # Pass every dirty image into the model, 
            # and compare with the clean image using MSE to get loss
            outputs = self._model(dirty_image)
            loss = self._criterion(outputs, clean_image)

            # backpropagation
            loss.backward()

            # update the parameters and keep track of the total loss
            self._optimizer.step()
            running_loss += loss.item()
    
        # Average the loss for this epoch and return it
        train_loss = running_loss / len(self._trainloader.dataset)

        return train_loss


    # Run model against a validation set to test generalizability.
    def _validate(self, epoch):
        # Put the model in evaluation mode
        self._model.eval()
        running_loss = 0.0

        # no_grad disables gradient calculation to reduce memory
        # consumption.
        with torch.no_grad():
            n_iter = len(self._valloader.dataset) // self._valloader.batch_size

            data = None
            for data in tqdm(self._valloader, total=n_iter):
                dirty_image = data[0].to(self._device)
                clean_image = data[1].to(self._device)

                # Pass the dirty image into the model and compare with
                # the clean image.
                outputs = self._model(dirty_image)
                loss = self._criterion(outputs, clean_image)

                running_loss += loss.item()

            # If finishing the first epoch, save the dirty and clean
            # images.
            # For example: output/saved_images/clean<epoch#>.png
            if data is not None:
                if epoch == 0:
                    save_image(dirty_image.cpu().data,
                               f'{self._output_dir}/dirty.png')
                    save_image(clean_image.cpu().data,
                               f'{self._output_dir}/clean.png')

                # Save the last clean and dirty image pair into outputs directory at the end of each epoch
                save_image(outputs.cpu().data,
                           f'{self._output_dir}/cleaned{epoch}.png')
        
            # Calculate the average loss for this epoch and return it
            val_loss = running_loss / len(self._valloader.dataset)
        
            return val_loss


    def train(self):
        # Record total training and validation losses across all epochs
        train_loss = []
        val_loss = []

        start = time.time()

        # Train the model and check against the validation set at each
        # epoch
        for epoch in range(self._epochs):
            print(f'Epoch {epoch+1} of {self._epochs}')

            # Train on the clean and dirty images in the training set
            train_epoch_loss = self._fit()
            print(f'Train Loss: {train_epoch_loss:.5f}')

            # Validate on the clean and dirty images in the validation
            # set
            val_epoch_loss = self._validate(epoch)
            print(f"Val Loss: {val_epoch_loss:.5f}")

            # Record average loss for the training and validation sets
            train_loss.append(train_epoch_loss)
            val_loss.append(val_epoch_loss)

            # Adjust the learning rate
            self._scheduler.step(val_epoch_loss)

        end = time.time()
        print(f'Took {((end - start) / 60):.3f} minutes to train')

        # Plot results
        plt.figure(figsize=(10, 7))
        plt.plot(train_loss, color='orange', label='train loss')
        plt.plot(val_loss, color='red', label='validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('output/loss.png')
        plt.show()

        print('Saving model...')
        torch.save(self._model.state_dict(), 'output/model.pth')
