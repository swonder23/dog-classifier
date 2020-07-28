# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 22:36:44 2020

@author: Steven
"""

##########################
## Step 0: Import Datasets
##########################

import numpy as np
from glob import glob

# load filenames for human and dog images
human_files = np.array(glob("data\lfw\*\*"))
dog_files = np.array(glob("data\dog_images\*\*\*"))

# print number of images in each dataset
print('There are %d total human images.' % len(human_files))
print('There are %d total dog images.' % len(dog_files))




##########################
## Step 1: Detect Humans
##########################

import cv2                
import matplotlib.pyplot as plt                                                    

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[0])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()


# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


### (IMPLEMENTATION) Assess the Human Face Detector ###

from tqdm import tqdm

human_files_short = human_files[:100]
dog_files_short = dog_files[:100]

nb_faces_actual_human = len(human_files_short)
nb_faces_actual_dog = 0    
nb_faces_detected_human = 0
nb_faces_detected_dog = 0

# perfect accuracy for human dataset would be 100%
for human in human_files_short:
    nb_faces_detected_human += face_detector(human)
accuracy_human = nb_faces_detected_human / nb_faces_actual_human 
print('The percentage of images in human_files_short having a detected face is ' + str(accuracy_human) + '.')
    
# perfect accuracy for dog dataset would be 0%   
for dog in dog_files_short:
    nb_faces_detected_dog += face_detector(dog)
accuracy_dog = nb_faces_detected_dog / (100 - nb_faces_actual_dog)
print('The percentage of images in dog_files_short having a detected face is ' + str(accuracy_dog) + '.')




##########################
## Step 2: Detect Dogs
##########################   
 
import torch
import torchvision.models as models

# define VGG16 model
VGG16 = models.vgg16(pretrained=True)

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# move model to GPU if CUDA is available
if use_cuda:
    VGG16 = VGG16.cuda()
    
    
from PIL import Image
import torchvision.transforms as transforms
        
#################################################################
## development section

# from https://pytorch.org/docs/stable/torchvision/models.html
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

## vgg16 expect 224x224 images
data_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

# load image
im = Image.open(dog_files[0])

## transform image
# https://stackoverflow.com/questions/57237381/runtimeerror-expected-4-dimensional-input-for-4-dimensional-weight-32-3-3-but
# vgg16 est fait pour recevoir des batches d'images. Comme on a une seule image, le batch_size ou n_samples est egal a 1. 
# n_samples est le premier element dans la shape. Utiliser unsqueeze(dim=0) ajoute une dimension a l'index 0
im_input = data_transform(im).unsqueeze(0)

# have to move image tensor to GPU: important since we moved vgg16 to GPU earlier
if use_cuda:
    im_input = im_input.cuda()
output = VGG16(im_input)
test = output.argmax().cpu().numpy().item()

## end development section
#################################################################


# we have now all the parts to build the function
def VGG16_predict(img_path):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path
    
    Args:
        img_path: path to an image
        
    Returns:
        Index corresponding to VGG-16 model's prediction
    '''
    
    ## TODO: Complete the function.
    ## Load and pre-process an image from the given img_path
    ## Return the *index* of the predicted class for that image
    
    # Load image
    im = Image.open(img_path)    
    
    # Transform image
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    data_transform = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            normalize,
        ])
    im_input = data_transform(im).unsqueeze(0)
    
    # have to move image tensor to GPU: important since we moved vgg16 to GPU earlier
    if use_cuda:
        im_input = im_input.cuda()
    pred = VGG16(im_input)
    cat_idx = pred.argmax().cpu().numpy().item()
    
    return cat_idx # predicted class index

VGG16_predict(dog_files_short[0])
VGG16_predict(dog_files_short[50])

### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    ## TODO: Complete the function.
    
    # dogs appear in an uninterrupted sequence and correspond to dictionary keys 151-268, inclusive
    is_dog = 151 <= VGG16_predict(img_path) <= 268
    
    return is_dog # true/false

dog_detector(dog_files_short[0])
dog_detector(dog_files_short[50])


### TODO: Test the performance of the dog_detector function
### on the images in human_files_short and dog_files_short.
nb_dogs_actual_human = 0
nb_dogs_actual_dog = len(dog_files_short)
nb_dogs_detected_human = 0
nb_dogs_detected_dog = 0

# perfect accuracy for human dataset would be 0%
for human in human_files_short:
    nb_dogs_detected_human += dog_detector(human)
accuracy_human = nb_dogs_detected_human / (100 - nb_dogs_actual_human) 
print('The percentage of images in human_files_short having a detected dog is ' + str(accuracy_human) + '.')
    
# perfect accuracy for dog dataset would be 100%   
for dog in dog_files_short:
    nb_dogs_detected_dog += dog_detector(dog)
accuracy_dog = nb_dogs_detected_dog / nb_dogs_actual_dog
print('The percentage of images in dog_files_short having a detected dog is ' + str(accuracy_dog) + '.')



##############################################################
## Step 3: Create a CNN to Classify Dog Breeds (from Scratch)
##############################################################

import os
from torchvision import datasets

### TODO: Write data loaders for training, validation, and test sets
## Specify appropriate transforms, and batch_sizes

# define training and test data directories
data_dir = 'data/dog_images'
train_dir = os.path.join(data_dir, 'train/')
valid_dir = os.path.join(data_dir, 'valid/')
test_dir = os.path.join(data_dir, 'test/')


# load and transform data using ImageFolder
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    
# convert data to a normalized torch.FloatTensor
# inspiration: https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101
train_test_transform = transforms.Compose([
    transforms.RandomResizedCrop(224), # randomly flip and rotate
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
    ])

# using other transformations for validation increase images variability and will decrease overfitting likeliness (i think?)
valid_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
    ])


train_data = datasets.ImageFolder(train_dir, transform=train_test_transform)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transform)
test_data = datasets.ImageFolder(test_dir, transform=train_test_transform)


# define dataloader parameters
## understanding relation between batch_size vs batch_length (basically batch length = number of images in full data / batch size)
## batch length is not defined by the user; batch size is
## https://discuss.pytorch.org/t/about-the-relation-between-batch-size-and-length-of-data-loader/10510
batch_size = 20
num_workers = 0 

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                           num_workers=num_workers, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, 
                                          num_workers=num_workers, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
                                          num_workers=num_workers, shuffle=True)





# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy() # convert images to numpy for display
len(labels)


### DONT'WORK ############################
# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
# display 20 images
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    plt.imshow(images[idx])
    ax.set_title(labels[idx])
#########################################











## (IMPLEMENTATION) Model Architecture
# nn.Conv2d(3, 16, 3, padding=1)
# nn.Conv2d(depth of input, nb feature maps/filters, kernel size (size of each filter), padding)
# padding=1: Note that here 1 row or column is padded on either side, so a total of 2
# rows or columns are added
# formula to get output volume
# (W−F+2P)/S+1 

import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net(nn.Module):
    ### TODO: choose an architecture, and complete the class
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  ## padding=1 will get us an output with the same size as input
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # linear layer (128 * 14 * 14 -> 10000)
        self.fc1 = nn.Linear(128 * 14 * 14, 10000)
        # linear layer (10000 -> 1000)
        self.fc2 = nn.Linear(10000, 1000)
        # linear layer (1000 -> 133)
        self.fc3 = nn.Linear(1000, 133)
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        
        # convolutional layer (sees 224x224x3 image tensor)
        x = self.pool(F.relu(self.conv1(x)))
        # convolutional layer (sees 112x112x16 tensor)
        x = self.pool(F.relu(self.conv2(x)))
        # convolutional layer (sees 56x56x32 tensor)
        x = self.pool(F.relu(self.conv3(x)))
        # convolutional layer (sees 28x28x64 tensor)
        x = self.pool(F.relu(self.conv4(x)))
        # flatten image input
        x = x.view(-1, 128 * 14 * 14)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = F.relu(self.fc2(x))
        # add dropout layer
        x = self.dropout(x)
        # add 3rd and final hidden layer
        x = self.fc3(x)
        return x

#-#-# You so NOT have to modify the code below this line. #-#-#

# instantiate the CNN
model_scratch = Net()

# move tensors to GPU if CUDA is available
if use_cuda:
    model_scratch.cuda()


## (IMPLEMENTATION) Specify Loss Function and Optimizer
import torch.optim as optim

### TODO: select loss function
criterion_scratch = nn.CrossEntropyLoss()

### TODO: select optimizer
optimizer_scratch = optim.SGD(model_scratch.parameters(), lr=0.01)

# dir(train_loader)
loaders_scratch = {
  "train": train_loader,
  "valid": valid_loader,
  "test": test_loader
}


## (IMPLEMENTATION) Train and Validate the Model
def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            ## The losses are averaged across observations for each minibatch
            ## because at the end, we will want to compute an average across all minibatches, we need to "unaverage" the loss here
            train_loss += loss.item()*data.size(0)
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss 
            ## because at the end, we will want to compute an average across all minibatches, we need to "unaverage" the loss here
            valid_loss += loss.item()*data.size(0)
            
        # calculate average losses
        train_loss = train_loss/len(loaders['train'].sampler)
        valid_loss = valid_loss/len(loaders['valid'].sampler)

            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss
            
    # return trained model
    return model


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
## had this error: OSError: image file is truncated (150 bytes not processed)
## fixed with suggested workaround: https://github.com/eriklindernoren/PyTorch-YOLOv3/issues/162

# train the model
model_scratch = train(5, loaders_scratch, model_scratch, optimizer_scratch, 
                      criterion_scratch, use_cuda, 'model_scratch.pt')

# load the model that got the best validation accuracy
model_scratch.load_state_dict(torch.load('model_scratch.pt'))




##############################################

### manual testing zone --> forward pass        
        
train_loss = 0.0

###################
# train the model #
###################
model_scratch.train()

dataiter = iter(loaders_scratch['train'])
data, target = dataiter.next()
    
# clear the gradients of all optimized variables
optimizer_scratch.zero_grad()
# forward pass: compute predicted outputs by passing inputs to the model
output = model_scratch(data)
# calculate the batch loss
loss = criterion_scratch(output, target)
# backward pass: compute gradient of the loss with respect to model parameters
loss.backward()
# perform a single optimization step (parameter update)
optimizer_scratch.step()
# update training loss
train_loss += loss.item()*data.size(0)

    
## https://discuss.pytorch.org/t/how-can-my-net-produce-negative-outputs-when-i-use-relu/19483/3
## use F.softmax(x, dim=1)
## in the testing section, softmax isn't applied before returning a breed (selon pred=.. below)
## to do: test if it gives the same thing applying softmax or not beforehand. I guess it does..
pred = output.data.max(1, keepdim=True)[1]

##############################################



#########################################################################
## Step 4: Create a CNN to Classify Dog Breeds (using Transfer Learning)
#########################################################################

import torchvision.models as models
import torch.nn as nn

# Load the pretrained ResNet-50 model from pytorch
resnet50 = models.resnet50(pretrained=True)

# print out the model structure
print(resnet50)
# print classifier layer (last full connected layer) number of input and output
print(resnet50.fc.in_features) 
print(resnet50.fc.out_features) 

## inspiration
## https://github.com/mortezamg63/Accessing-and-modifying-different-layers-of-a-pretrained-model-in-pytorch

child_counter = 0
for child in resnet50.children():
   print(" child", child_counter, "is:")
   print(child)
   child_counter += 1

# freezing training for all but last layer
ct = 0
for child in resnet50.children():
    ct += 1
    if ct < 9: ## before final linear layer
        for param in child.parameters():
            param.requires_grad = False

n_inputs = resnet50.fc.in_features        
# add last linear layer (n_inputs -> 133 dog breed classes)
# new layers automatically have requires_grad = True
last_layer = nn.Linear(n_inputs, 133)   
resnet50.fc = last_layer     
print(resnet50)

if use_cuda:
    model_transfer = resnet50.cuda()
    

criterion_transfer = nn.CrossEntropyLoss()
optimizer_transfer = optim.SGD(model_transfer.fc.parameters(), lr=0.01)


# train the model

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item()*data.size(0)
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss 
            valid_loss += loss.item()*data.size(0)
            
        # calculate average losses
        train_loss = train_loss/len(loaders['train'].sampler)
        valid_loss = valid_loss/len(loaders['valid'].sampler)

            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss
            
    # return trained model
    return model


# train the model
model_transfer = train(3, loaders_scratch, model_transfer, optimizer_transfer, 
                      criterion_transfer, use_cuda, 'model_transfer.pt')

# load the model that got the best validation accuracy
# model_transfer.load_state_dict(torch.load('model_transfer.pt'))

# test the model
test(loaders_scratch, model_transfer, criterion_transfer, use_cuda)



# list of class names by index, i.e. a name can be accessed like class_names[0]
asdf = [item[4:].replace("_", " ") for item in train_data.classes]
len(asdf)
