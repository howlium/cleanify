import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import cv2
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import argparse

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=40,
            help='number of epochs to train the model for')
args = vars(parser.parse_args())

# helper functions
def save_decoded_image(img, name):
    img = img.view(img.size(0), 3, 224, 224)
    save_image(img, name)
    
input_dir  = '../input'
output_dir = '../output'
image_dir  = output_dir + '/saved_images'
os.makedirs(image_dir, exist_ok=True)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)
batch_size = 2

dirty_files = os.listdir(f'{input_dir}/dirty')
if '.DS_Store' in dirty_files:
    dirty_files.remove('.DS_Store') # For macOS: Skip invisible Desktop Services Store file.
dirty_files.sort()

clean_files = os.listdir(f'{input_dir}/clean')
if '.DS_Store' in clean_files:
    clean_files.remove('.DS_Store') # For macOS: Skip invisible Desktop Services Store file.
clean_files.sort()

x_dirty = []
for i in range(len(dirty_files)):
    x_dirty.append(dirty_files[i])
y_clean = []
for i in range(len(clean_files)):
    y_clean.append(clean_files[i])

(x_train, x_val, y_train, y_val) = train_test_split(x_dirty, y_clean, test_size=0.25)
print(f"Train data instances: {len(x_train)}")
print(f"Validation data instances: {len(x_val)}")

# define transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

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

train_data = CleanDataset(x_train, y_train, transform)
val_data = CleanDataset(x_val, y_val, transform)
 
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

class CleanerCNN(nn.Module):
    def __init__(self):
        super(CleanerCNN, self).__init__()
        self.conv1 = nn.Conv2d(3,  64, kernel_size=9, padding=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=2)
        self.conv3 = nn.Conv2d(32,  3, kernel_size=5, padding=2)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.conv3(x)

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
