#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
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

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
image_datasets = dict()
image_datasets['train'] = datasets.ImageFolder(train_dir, transform=train_transforms)
image_datasets['valid'] = datasets.ImageFolder(valid_dir, transform=valid_transforms)
image_datasets['test'] = datasets.ImageFolder(test_dir, transform=test_transforms)


# TODO: Using the image datasets and the trainforms, define the dataloaders

dataloaders = dict()
dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=16, shuffle=True)
dataloaders['valid'] = torch.utils.data.DataLoader(image_datasets['valid'], batch_size=16)
dataloaders['test']  = torch.utils.data.DataLoader(image_datasets['test'], batch_size=16)

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

def pretrained_model(model_nm):
    model = None
    if model_nm == "vgg16":
        model = models.vgg16(pretrained=True)
    if model_nm == "vgg19":
        model = models.vgg19(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    return model
#model = models.vgg16(pretrained=True)
#model.name = "vgg16"

# Freezing parameters
#for param in model.parameters():
 #   param.requires_grad = False


def train_model(model, trainloader, epochs, freq, criterion, optimizer, device='cpu'):

    steps = 0
    j = freq
    model.to('cuda')

    for e in range(epochs):
        cum_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            cum_loss += loss.item()

            if steps % j == 0:
                model.eval()
                v_loss = 0
                v_acc=0
                for ii, (inputs2,labels2) in enumerate(dataloaders['valid']):
                    optimizer.zero_grad()
                    inputs2, labels2 = inputs2.to('cuda') , labels2.to('cuda')
                    model.to('cuda')
                    with torch.no_grad():
                        outputs = model.forward(inputs2)
                        v_loss = criterion(outputs,labels2)
                        ps = torch.exp(outputs).data
                        equality = (labels2.data == ps.max(1)[1])
                        v_acc += equality.type_as(torch.FloatTensor()).mean()

                v_loss = v_loss / len(dataloaders['valid'])
                v_acc = v_acc /len(dataloaders['valid'])

                print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Training Loss: {:.4f}".format(cum_loss/j),
                  "Validation Loss {:.4f}".format(v_loss),
                   "Accuracy: {:.4f}".format(v_acc))
                cum_loss = 0
parser = argparse.ArgumentParser(description="Training a model to predict images")
parser.add_argument("-l", "--learn_rate", type=float, default=0.001, help="Learning rate")
parser.add_argument("-e", "--epoch", type=int, default=3, help="number of iterations to perform during the training")
parser.add_argument("-g", "--gpu", help="GPU", default=False, action="store_true")
parser.add_argument("-a", "--arch", type=str, default="vgg16", help="What pretrained model to use. vgg16 or vgg19")
args = parser.parse_args()

print("Arguments used for training:")
print("GPU: " + str(args.gpu))
print("epochs: " + str(args.epoch))
print("Learning Rate: " + str(args.learn_rate))
print("Model Name: " + args.arch)

model = pretrained_model(args.arch)
#Using ReLU activations and dropout create a new untrained feed-forward network as a classifier.

num_feats = model.classifier[0].in_features

classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(num_feats, 512)),
                              ('relu', nn.ReLU()),
                              ('drpot', nn.Dropout(p=0.5)),
                              ('hidden', nn.Linear(512, 100)),
                              ('fc2', nn.Linear(100, 102)),
                              ('output', nn.LogSoftmax(dim=1)),
                              ]))

model.classifier = classifier

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learn_rate)
model.cuda()


#Train the model
train_model(model, dataloaders['train'],args.epoch, 50, criterion, optimizer, args.gpu)

# Do validation on the test set
tot_cnt = 0
correct_cnt = 0

#with torch.no_grad():
#    model.eval()
#   for data in dataloaders['test']:
#        img, lbl = data
#        img, lbl = images.to('cuda'), labels.to('cuda')
#        outputs = model(img)
#        _, pred = torch.max(outputs.data, 1)
#        tot_cnt += lbl.size(0)
#        correct_cnt += (pred == lbl).sum().item()

#print('Test Accuracy : %d%%' % (100 * correct_cnt / tot_cnt))

#  Save the checkpoint
model.class_to_idx = dataloaders['train'].dataset.class_to_idx
#model.epochs = 3
checkpoint = {'input_size': [3, 224, 224],
                 'batch_size': dataloaders['train'].batch_size,
                  'output_size': 102,
                  'state_dict': model.state_dict(),
                  'optimizer_dict':optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'epoch': args.epoch,
                  'model_name':args.arch,
                  'classifier':model.classifier}
torch.save(checkpoint, 'my_checkpoint.pth')
