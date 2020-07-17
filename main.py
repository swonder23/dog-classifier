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

# perfect accuracy for human dataset should be 100%
for human in human_files_short:
    nb_faces_detected_human += face_detector(human)
accuracy_human = nb_faces_detected_human / nb_faces_actual_human 
print(accuracy_human)
    
# perfect accuracy for dog dataset should be 0%   
for dog in dog_files_short:
    nb_faces_detected_dog += face_detector(dog)
accuracy_dog = nb_faces_detected_dog / (100 - nb_faces_actual_dog)
print(accuracy_dog)




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
        

## development section
#################################################################

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

