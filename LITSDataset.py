import os
import numpy as np
import math
from PIL import Image, ImageOps
from torch.utils.data import Dataset

class LitsDataset(Dataset):
  CLASSES = ['background','liver','tumor']

  def __init__(self, imgs_dir, masks_dir, classes=None, augmentation=None, preprocessing=None):
    self.ids = []
    self.images_fps = []
    self.masks_fps = []

    self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

    self.augmentation = augmentation
    self.preprocessing = preprocessing

    for i in imgs_dir:
      self.ids = self.ids + os.listdir(i)

    for i in imgs_dir:
      for j in os.listdir(i):
        self.images_fps.append(os.path.join(i, j))
    

    for i in masks_dir:
      for j in os.listdir(i):
        self.masks_fps.append(os.path.join(i, j))
    

  def __len__(self):
    return len(self.ids)

  def __getitem__(self, idx):
    image = Image.open(self.images_fps[idx])
    image = ImageOps.grayscale(image)
    image = np.array(image)

    mask = Image.open(self.masks_fps[idx])
    mask_array = np.array(mask)

    mask_array = (np.where(mask_array==127, 1, mask_array))
    mask = (np.where(mask_array==255, 2, mask_array)).astype('float')


    # masks = [(mask == v) for v in self.class_values]
    # mask = np.stack(masks, axis=-1).astype('float')


    # # add background if mask is not binary
    # if mask.shape[-1] != 1:
    #     background = 1 - mask.sum(axis=-1, keepdims=True)
    #     mask = np.concatenate((mask, background), axis=-1)
    #     mask = mask.astype(np.uint8) 
        
    # mask = (np.where(mask==1, 255, mask))

    # apply augmentations
    if self.augmentation:
        sample = self.augmentation(image=image, mask=mask)
        image, mask = sample['image'], sample['mask']

    # apply preprocessing
    if self.preprocessing:
        # sample = self.preprocessing(image=image, mask=mask)
        # image, mask = sample['image'], sample['mask']
        image = self.preprocessing(image)
        mask = self.preprocessing(mask)

    return image, mask[0]