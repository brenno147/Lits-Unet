from util import CCEIoULoss
from TTVSpliter import defineTrainTestValidation
from LITSDataset import LitsDataset
from torchvision import transforms
from segmentation_models_pytorch.losses import JaccardLoss
import segmentation_models_pytorch as smp
import torchmetrics
from torch import nn
import torch
import numpy as np
from PIL import Image
import os

trainPercents =  70
testPercents = 20
validationPercents = 10
x_train_dir,y_train_dir,x_test_dir,y_test_dir,x_valid_dir,y_valid_dir = defineTrainTestValidation(trainPercents,testPercents,validationPercents)

CLASSES = ["Liver","Tumor"]

transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256), interpolation=Image.NEAREST)])

train_dataset = LitsDataset(x_train_dir, y_train_dir, classes=CLASSES, preprocessing=transform)
valid_dataset = LitsDataset(x_valid_dir, y_valid_dir, classes=CLASSES, preprocessing=transform)
test_dataset = LitsDataset(x_test_dir, y_test_dir, classes=CLASSES, preprocessing=transform)

x = torch.tensor([[1,1,1,1], [0,2,2,2]])

print(nn.functional.one_hot(x, 3))