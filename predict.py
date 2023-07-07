import matplotlib.pyplot as plt
import numpy as np
from LITSDataset import LitsDataset
from TTVSpliter import defineTrainTestValidation
from torchvision import transforms
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image, ImageOps
import torch
import os


def prepare_plot(origImage, origMask1, origMask2, predMask1, predMask2):
    # initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=5, figsize=(10, 10))
    # plot the original image, its mask, and the predicted mask
    ax[0].imshow(origImage)
    ax[1].imshow(origMask1)
    ax[2].imshow(origMask2)
    ax[3].imshow(predMask1)
    ax[4].imshow(predMask2)
    # set the titles of the subplots
    ax[0].set_title("Image")
    ax[1].set_title("Original Mask")
    ax[2].set_title("Original Mask 2")
    ax[3].set_title("Predicted Mask")
    ax[4].set_title("Predicted Mask 2")
    # set the layout of the figure and display it
    figure.tight_layout()
    plt.savefig("results/predict.png")


trainPercents =  70
testPercents = 20
validationPercents = 10
x_train_dir,y_train_dir,x_test_dir,y_test_dir,x_valid_dir,y_valid_dir = defineTrainTestValidation(trainPercents,testPercents,validationPercents)

transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize((256, 256)),transforms.ToTensor()])

test_dataset = LitsDataset(x_test_dir, y_test_dir, classes=["Liver", "Tumor"], preprocessing=transform)

unet = torch.load("results/unet.pth")
unet.cuda()

img, y = test_dataset[30]
x = img.cuda()

x = x[None, :, :, :]

unet.eval()

with torch.no_grad():
    pred = unet(x)
    pred = pred.cpu().numpy()


pred = np.array(pred)

ground = y.numpy()

print(ground.shape)

prepare_plot(img[0], ground[0], ground[1], pred[0][0], pred[0][1])