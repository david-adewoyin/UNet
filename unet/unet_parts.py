import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

class DoubleConv(keras.layers.Layer):

  """ (convolution => [BN] => Relu) *2 
  Each step in the contracting and expansive path have two 3x3 convolutional layer
  In the U-Nnet padding they used 0 padding but we use 1 padding so that we don't crop the final feature map
  """
  def __init__(self,filters):
    super(DoubleConv,self).__init__()

    # First 3x3 convolutional layer
    self.first = layers.Conv2D(filters, kernel_size = 3, padding='same', kernel_initializer = 'he_normal')
    self.batch1 = layers.BatchNormalization()
    self.act1 = layers.Activation('relu')

    # 2nd 3x3 convolutional layer
    self.second= layers.Conv2D(filters, kernel_size = 3, padding='same', kernel_initializer = 'he_normal')
    self.batch2 = layers.BatchNormalization()
    self.act2 = layers.Activation('relu')

  def call(self,inputs):
    # Apply the two convolutional layers
    x = self.first(inputs)
    x = self.batch1(x)
    x = self.act1(x)

    x = self.second(x)
    x = self.batch2(x)
    return self.act2(x)

class DownSampling(keras.layers.Layer):
  """ downsamples the feature map with a maxpool layer(2) followed by a double conv """

  def __init__(self,filters):
    super(DownSampling,self).__init__()
    # Max pooling Layer
    self.pool = layers.MaxPooling2D(2)
    self.conv = DoubleConv(filters)

  def call(self,inputs):
    x = self.pool(inputs)
    return self.conv(x)

class UpSampling(keras.layers.Layer):
  """Each Step in the expansive path up-samples the feature map with a 2x2 convolution follwed a double conv"""

  def __init__(self,filters):
    super(UpSampling,self).__init__()
    self.up = layers.Conv2DTranspose(filters, kernel_size = 2, strides = 2, padding = 'same', kernel_initializer = 'he_normal')
    self.conv = DoubleConv(filters)
  
  def call(self,inputs,corresponding_map):
   upsamp = self.up(inputs)
 
   crop = layers.CenterCrop(upsamp.shape[1],upsamp.shape[2])(corresponding_map)
   x = layers.Concatenate()([upsamp,crop])
   return self.conv(x)