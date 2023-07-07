from TTVSpliter import defineTrainTestValidation
from LITSDataset import LitsDataset
from model.ConvMixUnet import UNetModel1
from Trainer import LitAutoEncoder
from torch.utils.data import DataLoader
from torchvision import transforms
import segmentation_models_pytorch as smp
import os
from PIL import Image
from util import get_training_augmentation
import lightning.pytorch as pl
from pytorch_lightning.loggers import WandbLogger

BASE_OUTPUT = "results"
MODEL_PATH = os.path.join(BASE_OUTPUT, "unet.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

epochs = 3

batch_size = 8

trainPercents =  70
testPercents = 20
validationPercents = 10
x_train_dir,y_train_dir,x_test_dir,y_test_dir,x_valid_dir,y_valid_dir = defineTrainTestValidation(trainPercents,testPercents,validationPercents)

CLASSES = ["Liver","Tumor"]

transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=Image.NEAREST)])
augmentations = get_training_augmentation()

train_dataset = LitsDataset(x_train_dir, y_train_dir, classes=CLASSES, preprocessing=transform, augmentation=augmentations)
valid_dataset = LitsDataset(x_valid_dir, y_valid_dir, classes=CLASSES, preprocessing=transform)
test_dataset = LitsDataset(x_test_dir, y_test_dir, classes=CLASSES, preprocessing=transform)

trainLoader = DataLoader(train_dataset, batch_size=batch_size, num_workers=os.cpu_count())
validLoader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=os.cpu_count())
testLoader = DataLoader(test_dataset, batch_size=batch_size, num_workers=os.cpu_count())

unet = smp.Unet(
    encoder_name="resnet18",        
    encoder_weights="imagenet",    
    in_channels=1,                  
    classes=3
)


wandb_logger = WandbLogger(project='Lits-Unet')
wandb_logger.experiment.config["batch_size"] = batch_size

trainer = pl.Trainer(default_root_dir="results/unetResnet18", callbacks=[pl.callbacks.EarlyStopping(monitor="valid_loss", mode="min")], max_epochs=50, logger=wandb_logger)
autoencoder = LitAutoEncoder(unet)
trainer.fit(model=autoencoder, train_dataloaders=trainLoader, val_dataloaders=validLoader)
trainer.test(model=autoencoder, dataloaders=testLoader)