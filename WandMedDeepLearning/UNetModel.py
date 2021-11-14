import torch 
import torch.nn as nn 
import numpy as np
import torch.nn.functional as F

class UNet(nn.Module):
    """A class for the convolutional neural network UNet. 

    """

    def __init__(self):
      """Instantiation method for UNet. 

      """

      super(UNet, self).__init__()
      self.maxPool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
      self.downConv1 = DoubleConv(1,64)
      self.downConv2 = DoubleConv(64,128)
      self.downConv3 = DoubleConv(128,256)
      self.downConv4 = DoubleConv(256, 512)
      self.downConv5 = DoubleConv(512,1024)
      self.upTrans1 = nn.ConvTranspose2d(in_channels=1024, out_channels = 512, kernel_size = 2, stride = 2)
      self.upConv1 = DoubleConv(1024, 512)
      self.upTrans2 = nn.ConvTranspose2d(in_channels=512, out_channels = 256, kernel_size = 2, stride = 2)
      self.upConv2 = DoubleConv(512, 256)
      self.upTrans3 = nn.ConvTranspose2d(in_channels=256, out_channels = 128, kernel_size = 2, stride = 2)
      self.upConv3 = DoubleConv(256, 128)
      self.upTrans4 = nn.ConvTranspose2d(in_channels=128, out_channels = 64, kernel_size = 2, stride = 2)
      self.upConv4 = DoubleConv(128, 64)
      self.out = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)

    def forward(self, image):
        """Performs the U architecture (encoder and decoder) of UNet.

        Args:
            image (2D tensor): the input image
            
        Returns:
            x (2D tensor): the segmentation mask

        """

        #bs, c , h ,w
        #The encoder (first half of "U")
        x1 = self.downConv1(image) #
        x2 = self.maxPool_2x2(x1)
        x3 = self.downConv2(x2) #
        x4 = self.maxPool_2x2(x3)
        x5 = self.downConv3(x4)  #
        x6 = self.maxPool_2x2(x5)
        x7 = self.downConv4(x6)  #
        x8 = self.maxPool_2x2(x7)
        x9 = self.downConv5(x8)
        #Now the decoder part (second half of "U")CA

        #concatenate with x7, but first need to crop x7.
        x = self.upTrans1(x9)       
        y = Crop(x7, x)
        x = self.upConv1(torch.cat([x, y], 1))

        x = self.upTrans2(x)       
        y = Crop(x5, x)
        x = self.upConv2(torch.cat([x, y], 1))

        x = self.upTrans3(x)       
        y = Crop(x3, x)
        x = self.upConv3(torch.cat([x, y], 1))

        x = self.upTrans4(x)       
        y = Crop(x1, x)
        x = self.upConv4(torch.cat([x, y], 1))
        x = self.out(x)    
        return x

    def trainingStep(self, x, y):
        """Performs a training step. 

        Args: 
            x (2D tensor): the CT image 
            y (2D tensor): the ground truth mask

        Returns: 
            loss (float): the binary cross entropy loss of the model 
                using training data

        """

        out = self(x)
        loss = nn.BCEWithLogitsLoss()(out.float(), y.float())    #
        return loss

    def validationStep(self, x, y):
        """Performs a validation step. 

        Args: 
            x (2D tensor): the CT image 
            y (2D tensor): the ground truth mask

        Returns:
            loss (float): the binary cross entropy loss of the model 
                using validation data

        """

        out = self(x)
        loss = nn.BCEWithLogitsLoss()(out.float(), y.float())    #
        return loss
