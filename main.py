from __future__ import print_function

# import torch
import torch as torch
import torch.nn as nn
# import touch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy

device = torch.device("cpu")
#img size depending on if a cuda is availble
imsize = 128

loader = transforms.Compose([
    transforms.Resize(imsize),# scales image
    transforms.ToTensor() #transform image into a torch tensor
])

def image_loader(image_name):
    image = Image.open(image_name)
    #fake batch dimensions required to fit neworks input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

style_img = image_loader("./images/picasso.jpg")
content_img = image_loader("./images/dancing.jpg")

assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"

unloader = transforms.ToPILImage() #recovert the file into a PIL image

plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone() # we clone the tensor duh
    image = image.squeeze(0) #removing fake dimensions
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) #pause for a bit so the plot is updated

#running func on style image
plt.figure()
imshow(style_img, title="Style Image")

#runnign func on content image
plt.figure() 
imshow(content_img, title="Content Image")

