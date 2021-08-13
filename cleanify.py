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

# Construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=40,
            help='number of epochs to train the model for')
parser.add_argument("-a", "--autoencoder", action="store_true",
            help="Use an autoencoder NN instead of a vanilla CNN")
parser.add_argument("-d", "--distortion", type=int, default=0,
            help="The type of distortion applied to the images")
# vars returns a _dict_ that is an attribute of the object created by parse_args()
args = vars(parser.parse_args())

distortion = args['distortion']

# Make strings for directories
input_dir  = 'input'
output_dir = 'output'
image_dir  = output_dir + '/saved_images_%i' % distortion

# Make the image directory, if it already exists no FileExistsError will be raised
os.makedirs(image_dir, exist_ok=True)

# Make a string for device depending on whether the system supports CUDA
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

# Used in trainloader and valloader, then by dataloader
# Would be nice to change this name cause it's the same as the arg name
batch_size = 2

# Make a list of the names of the entries in the 'dirty' directory
dirty_files = os.listdir(f'{input_dir}/dirty')

# Remove .DS_Store and sort the files
# For macOS: Skip invisible Desktop Services Store file.
if '.DS_Store' in dirty_files:
    dirty_files.remove('.DS_Store') 
dirty_files.sort()

clean_files = os.listdir(f'{input_dir}/clean')
if '.DS_Store' in clean_files:
    clean_files.remove('.DS_Store') # For macOS: Skip invisible Desktop Services Store file.
clean_files.sort()

# Make lists of each set -- dirty and clean images
x_dirty = []
for i in range(len(dirty_files)):
    x_dirty.append(dirty_files[i])
y_clean = []
for i in range(len(clean_files)):
    y_clean.append(clean_files[i])

# Split arrays or matrices into random train and test subsets
(x_train, x_val, y_train, y_val) = train_test_split(x_dirty, y_clean, test_size=0.25)
print(f"Train data instances: {len(x_train)}")
print(f"Validation data instances: {len(x_val)}")

# Use Torchvision Compose() to combine transforms into one call
# ToPILImage makes a Pillow image
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

# CleanDataset class reads in images, applies transforms if entered, and returns them
# Will do for dirty set, and also clean set if clean_paths is entered
# In use, it saves as a tensor a pillow image of size 224 x 224
# and does that to all training and validation sets: original and blurred
class CleanDataset(Dataset):
    def __init__(self, dirty_paths, clean_paths=None, transforms=None):
        self.X = dirty_paths
        self.y = clean_paths
        self.transforms = transforms
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        dirty_image = cv2.imread(f"{input_dir}/dirty/{self.X[i]}")
        
        if self.transforms:
            dirty_image = self.transforms(dirty_image)
            
        if self.y is not None:
            clean_image = cv2.imread(f"{input_dir}/clean/{self.y[i]}")
            clean_image = self.transforms(clean_image)
            return (dirty_image, clean_image)
        else:
            return dirty_image

# Format the images by creating two CleanDataset class instances
train_data = CleanDataset(x_train, y_train, transform)
val_data = CleanDataset(x_val, y_val, transform)
 
# Load the training and validation datasets, and shuffle
# Each dataset has a batch size of 2, one for original the other for blurred
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# The Convolutional Neural Network
# Performs the convolutions and activations in the order CRCRF
class CleanerCNN(nn.Module):

    def __init__(self):
        super(CleanerCNN, self).__init__()

        if args["autoencoder"]:
            # encoder
            self.enc1 = nn.Linear(in_features=64, out_features=128)
            self.enc2 = nn.Linear(in_features=128, out_features=64)
            self.enc3 = nn.Linear(in_features=64, out_features=32)
            self.enc4 = nn.Linear(in_features=32, out_features=16)
            # decoder
            self.dec1 = nn.Linear(in_features=16, out_features=32)
            self.dec2 = nn.Linear(in_features=32, out_features=64)
            self.dec3 = nn.Linear(in_features=64, out_features=128)
            self.dec4 = nn.Linear(in_features=128, out_features=64)

        else:
            # NB: torch.nn.Conv2d is different than torch.nn.functional.Conv2d
            # Parameters entered are in_channels=3, one for each: RGB,
            # out_channels=64,
            # the kernel is 9x9,
            # padding is 2 for both height and width.
            # So the filter for this convolution is a tensor shaped 9 x 9 x 3 x 64
            # In other words, 64 filters get applied to each of points in the 9x9 kernel,
            # over all 3 layers.
            # By convention, padding would be (k - 1)/2, so
            # for k=9, padding=4
            # for k=1, padding=0
            # for k=5, padding=2
            # One output is produced per filter.
            # The pixel value is multiplied by the filter's kernel's pixel value, then added
            # across the kernel, and added across the filters to produce a single value
            # for each pixel of the original (single layer) image for each channel layer.
            # Other parameter for Conv2d() are stride, padding_mode, dilation, groups, and bias.
            # It looks like conv2 is a bottleneck module?
            self.conv1 = nn.Conv2d(3,  64, kernel_size=5, padding=2, bias=True)
            self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
            self.conv3 = nn.Conv2d(32,  3, kernel_size=3, padding=1, bias=True)
            # self.conv3 = nn.Conv2d(32, 16, kernel_size=1, padding=0, bias=True)
            # self.conv4 = nn.Conv2d(16,  3, kernel_size=5, padding=2)

    # overrides Module class's forward()
    def forward(self, x):
        if args["autoencoder"]:
            x = F.relu(self.enc1(x))
            x = F.relu(self.enc2(x))
            x = F.relu(self.enc3(x))
            x = F.relu(self.enc4(x))

            x = F.relu(self.dec1(x))
            x = F.relu(self.dec2(x))
            x = F.relu(self.dec3(x))
            x = F.relu(self.dec4(x))
        else:
            x = F.leaky_relu(self.conv1(x))
            x = F.leaky_relu(self.conv2(x))
            x = self.conv3(x)
            # x = F.leaky_relu(self.conv3(x))
            # x = self.conv4(x)

        return x

# to() moves and/or casts the parameters and buffers
# device is cpu or cuda:0
# returns self
model = CleanerCNN().to(device)
print(model)

# the loss function: Mean Squared Error
criterion = nn.MSELoss()

# the optimizer holds the current state and updates the parameters
# lr is learning rate
# Adam is proposed in https://arxiv.org/abs/1412.6980
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Reduce the learning rate based on how training is going
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=5,
        factor=0.5,
        verbose=True
    )
    

# This function doesn't use the arg epoch. Can we delete it?
# Run through all the data once, returning the training loss for the epoch
# Called once per epoch with the training data set passed in as dataloader
def fit(model, dataloader):

    # Set the model in training mode
    model.train()
    running_loss = 0.0

    # i doesn't get used in this for loop
    # tqdm wraps around an interable to make a progress bar
    for data in tqdm(dataloader, total=int(len(train_data)/dataloader.batch_size)):
        dirty_image = data[0]
        clean_image = data[1]
        dirty_image = dirty_image.to(device)
        clean_image = clean_image.to(device)

        # Set the gradients of all optimized torch Tensors to zero
        optimizer.zero_grad()

        # Pass every dirty image into the model, 
        # and compare with the clean image using MSE to get loss
        outputs = model(dirty_image)
        loss = criterion(outputs, clean_image)

        # backpropagation
        loss.backward()

        # update the parameters and keep track of the total loss
        optimizer.step()
        running_loss += loss.item()
    
    # Average the loss for this epoch and return it
    train_loss = running_loss/len(dataloader.dataset)
    print(f"Train Loss: {train_loss:.5f}")

    return train_loss



# How's training going for this epoch
# Called once per epoch with the validation set passed in as dataloader
def validate(model, dataloader, epoch):
    
    # get the last index
    last = int((len(val_data)/dataloader.batch_size)-1)

    # Set the module in evaluation mode
    model.eval()
    running_loss = 0.0

    # no_grad disables gradient calculation to reduce memory consumption
    with torch.no_grad():

        # Use the progress bar
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            dirty_image = data[0]
            clean_image = data[1]
            dirty_image = dirty_image.to(device)
            clean_image = clean_image.to(device)

            # Pass the dirty image into the model
            # and compare with the clean image using MSE to get loss
            outputs = model(dirty_image)
            loss = criterion(outputs, clean_image)

            # Update the running loss
            running_loss += loss.item()

            # If finishing the first epoch save the clean and dirty image
            # for example: output/saved_images/clean<epoch#>.png
            if epoch == 0 and i > last - 9:
                save_image(clean_image.cpu().data, f"{image_dir}/clean_e{epoch}_i{i}_d{distortion}.png")
                save_image(dirty_image.cpu().data, f"{image_dir}/dirty_e{epoch}_i{i}_d{distortion}.png")
            
            # Save the last clean and dirty image pair into outputs directory at the end of each epoch
            if epoch == args['epochs'] - 1 and i > last - 9:
                save_image(outputs.cpu().data, f"{image_dir}/cleaned_e{epoch}_i{i}_d{distortion}.png")
        
        # Calculate the average loss for this epoch and return it
        val_loss = running_loss/len(dataloader.dataset)
        print(f"Val Loss: {val_loss:.5f}")
        
        return val_loss


# Record the training and validation losses for all the epochs
train_loss  = []
val_loss = []

# How long is this going to take
start = time.time()

# Train the model and check against the validation set at each epoch
for epoch in range(args['epochs']):
    print(f"Epoch {epoch+1} of {args['epochs']}")

    # Train on the clean and dirty images in the training set
    train_epoch_loss = fit(model, trainloader)

    # Validate on the clean and dirty images in the validation set
    val_epoch_loss = validate(model, valloader, epoch)

    # Record average loss for the training and validation sets
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)

    # Adjust the learning rate
    scheduler.step(val_epoch_loss)

# Stop the watch and report
end = time.time()
print(f"Took {((end-start)/60):.3f} minutes to train")

# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('output/loss.png')
plt.show()
# save the model to disk
print('Saving model...')
torch.save(model.state_dict(), 'output/model.pth')
