"""
Implementation of the Complete UNet Architecture
"""

from .unet_parts import *

class UNet(keras.Model):

  def __init__(self,num_classes=3):
      
    super(UNet,self).__init__()
    self.inc = DoubleConv(filters=64)

    self.down1 = DownSampling(filters=128)
    self.down2 = DownSampling(filters=256)
    self.down3 = DownSampling(filters=512)
    self.down4 = DownSampling(filters = 1024)

    self.up1 = UpSampling(filters = 1024)
    self.up2 = UpSampling(filters = 512)
    self.up3 = UpSampling(filters = 256)
    self.up4 = UpSampling(filters = 128)

    self.out = DoubleConv(filters=64)

    # final prediction classes
    self.outc = layers.Conv2D(num_classes, kernel_size = 1,padding='same')

  def call(self,inputs):
    skip1 = self.inc(inputs)
    skip2 = self.down1(skip1)
    skip3 = self.down2(skip2)
    skip4 = self.down3(skip3)

    x = self.down4(skip4)

    x = self.up1(x,skip1)
    x = self.up2(x,skip2)
    x = self.up3(x,skip3)
    x = self.up4(x,skip4)
    
    x = self.out(x)
    
    return self.outc(x)