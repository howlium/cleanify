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
# vars returns a _dict_ that is an attribute of the object created by parse_args()
args = vars(parser.parse_args())


# Helper function
# Save a tensor into an image file with PyTorch
def save_decoded_image(img, name):
    # view() returns a new tensor with the same data but different shape
    # size(0) returns the first dimension of the tensor img
    img = img.view(img.size(0), 3, 224, 224)
    save_image(img, name)


# Make strings for directories
input_dir  = '../input'
output_dir = '../output'
image_dir  = output_dir + '/saved_images'

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
    transforms.Resize((224, 224)),
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
        # It looks like conv2 is a bottleneck module?:w
        self.conv1 = nn.Conv2d(3,  64, kernel_size=9, padding=4, bias=True)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0, bias=True)
        self.conv3 = nn.Conv2d(32,  3, kernel_size=5, padding=2)
        # self.conv3 = nn.Conv2d(32, 16, kernel_size=1, padding=0, bias=True)
        # self.conv4 = nn.Conv2d(16,  3, kernel_size=5, padding=2)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.conv3(x)
        # x = F.leaky_relu(self.conv3(x))
        # x = self.conv4(x)

        return x

model = CleanerCNN().to(device)
print(model)

# the loss function
criterion = nn.MSELoss()

# the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=5,
        factor=0.5,
        verbose=True
    )
    
def fit(model, dataloader, epoch):
    model.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        dirty_image = data[0]
        clean_image = data[1]
        dirty_image = dirty_image.to(device)
        clean_image = clean_image.to(device)
        optimizer.zero_grad()
        outputs = model(dirty_image)
        loss = criterion(outputs, clean_image)
        # backpropagation
        loss.backward()
        # update the parameters
        optimizer.step()
        running_loss += loss.item()
    
    train_loss = running_loss/len(dataloader.dataset)
    print(f"Train Loss: {train_loss:.5f}")
    
    return train_loss
    
def validate(model, dataloader, epoch):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            dirty_image = data[0]
            clean_image = data[1]
            dirty_image = dirty_image.to(device)
            clean_image = clean_image.to(device)
            outputs = model(dirty_image)
            loss = criterion(outputs, clean_image)
            running_loss += loss.item()
            if epoch == 0 and i == int((len(val_data)/dataloader.batch_size)-1):
                save_decoded_image(clean_image.cpu().data, name=f"{image_dir}/clean{epoch}.png")
                save_decoded_image(dirty_image.cpu().data, name=f"{image_dir}/dirty{epoch}.png")
            if i == int((len(val_data)/dataloader.batch_size)-1):
                save_decoded_image(outputs.cpu().data, name=f"{image_dir}/cleaned{epoch}.png")
        val_loss = running_loss/len(dataloader.dataset)
        print(f"Val Loss: {val_loss:.5f}")
        
        return val_loss
        
train_loss  = []
val_loss = []
start = time.time()
for epoch in range(args['epochs']):
    print(f"Epoch {epoch+1} of {args['epochs']}")
    train_epoch_loss = fit(model, trainloader, epoch)
    val_epoch_loss = validate(model, valloader, epoch)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    scheduler.step(val_epoch_loss)
end = time.time()
print(f"Took {((end-start)/60):.3f} minutes to train")

# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('../output/loss.png')
plt.show()
# save the model to disk
print('Saving model...')
torch.save(model.state_dict(), '../output/model.pth')
