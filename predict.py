import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms, utils
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

import argparse

import copy
from PIL import Image
import matplotlib.pyplot as plt
import json
from collections import OrderedDict
import time
import os

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)

    prepoceess_img = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    py_img = prepoceess_img(img)
    # Converting to Numpy array
    array_img = np.array(py_img)

    # Converting to torch tensor
    tensor_img = torch.from_numpy(array_img).type(torch.FloatTensor)
    # Adding dimension to image
    img_ret = tensor_img.unsqueeze_(0)

    return img_ret

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

def pretrained_model(model_nm):
    model = None
    if model_nm == "vgg16":
        model = models.vgg16(pretrained=True)
    if model_nm == "vgg19":
        model = models.vgg19(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    return model

def predict(flower_img, model, topk, gpu_mode):


    print('####prediction starts####')
    model.eval()



    with torch.no_grad():
        # Running image through network
        output = model.forward(flower_img)

    # Calculating probabilities
    probs = torch.exp(output)
    probs_top = probs.topk(topk)[0]
    index_top = probs.topk(topk)[1]

    # Converting probabilities and outputs to lists
    probs_top_list = np.array(probs_top)[0]
    index_top_list = np.array(index_top[0])

    # Loading index and class mapping
    class_to_idx = model.class_to_idx
    # Inverting index-class dictionary
    inx_to_class = {x: y for y, x in class_to_idx.items()}

    # Converting index list to class list
    classes_top_list = []
    for index in index_top_list:
        classes_top_list += [inx_to_class[index]]

    return probs_top_list, classes_top_list


def load_model(filepath):
    print("#####Loading the checkpointed model####")

    load_data = torch.load(filepath)


    epochs = load_data['epoch']
    Optimizer = load_data['optimizer_dict']
    model_state = load_data['state_dict']
    model_name = load_data['model_name']
    classifier = load_data['classifier']

    model = pretrained_model(model_name)
    model.classifier = classifier

    model.class_to_idx = load_data['class_to_idx']
    model.load_state_dict(load_data['state_dict'])
    print("#####Loading finished ####")
    return model

ap = argparse.ArgumentParser(description='Predicting flower from image')

ap.add_argument('--img_in', default='./flowers/test/10/image_07090.jpg', nargs='?', action="store", type = str)
ap.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
ap.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
ap.add_argument('--gpu', default="gpu", action="store", dest="gpu")
ap.add_argument("-p", "--path", type=str, required=True, help="filepath and filename of saved model")
args = ap.parse_args()



model = load_model(args.path)

if args.gpu:
   model.to('cuda')
else:
   model.to('cpu')
flower_img = process_image(args.img_in)

if args.gpu:
    flower_img = flower_img.to('cuda')
else:
    pass
probs, classes = predict(flower_img, model, args.top_k, args.gpu)


print(probs)
print(classes)

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
flower_name = []
for i in classes:
    flower_name += [cat_to_name[i]]
print(f"predicted flower : '{flower_name[0]}' has a probability of {round(probs[0]*100,4)}% ")
