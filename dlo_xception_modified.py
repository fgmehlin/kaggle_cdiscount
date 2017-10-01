import os
import glob
from scipy import misc
import numpy as np
import keras.models as models
from keras import layers
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Activation, Dropout, Dense, BatchNormalization, Input, GlobalAveragePooling2D, AveragePooling2D, Conv2D, MaxPooling2D, SeparableConv2D
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm_notebook as tqdmn
from keras.callbacks import ModelCheckpoint


""" 
Xception from Francois Chollet. See https://arxiv.org/pdf/1610.02357.pdf for explanation of the model, links with Inception and Residual models. 

Implementation taken directly from https://github.com/fchollet/deep-learning-models/blob/master/xception.py

The Entry and Middle Flow are identical to the ones from the orignal network, as they act as general feature extractors for images. The dimensions of the Exit Flow module have been modified a bit to suit better the cdiscount dataset, as this module acts as a classifier for the given dataset/classes, based on the features extracted by the Entry and Middle Flow. 
The original network has been crafted for the Imagenet dataset having 1000 classes, Cdiscount's dataset has 5270 classes.

"""


img_width = 96 # Change to what you want
img_height = 96 # Change to what you want
n_channel = 3


"""
Slow, uncomment if you don't train on the full dataset

 class_list = os.listdir(train_folder) 
n_classes = len(class_list) 
"""

n_classes = 5270 # Comment if you don't train on the full dataset
img_input = Input(shape=(img_width,img_height,n_channel))


"""
Entry Flow. Takes input of shape (a,a,3) and outputs tensor of shape ~ (a/16,a/16,728).
Warning: a should not be too small. The overall network (Entry, Middle and Exit flows) transforms the input of shape (a,a,3) to ~ (a/32,a/32,5270).
a/32 should be at >= 3 because some convolutions with filters of shape (3,3) are computed on (a/32,a/32) in the end of the network.
The code will not crash if a/32 < 3, but it will not be optimal.
If you want to use this code on an input with a/32 < 3, it is better to change the code to reduce the number of times the input's spatial shape is reduced, by performing the commented instructions in the Entry flow for example. When performing these instructions, the overall network will reduce the shape from (a,a,3) to ~ (a/16, 1/16, 5270) instead of (a/32, a/32, 5270). Both instructions have to be performed simultaneously.
"""

x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, name='block1_conv1')(img_input)
x = BatchNormalization(name='block1_conv1_bn')(x)
x = Activation('relu', name='block1_conv1_act')(x)
x = Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
x = BatchNormalization(name='block1_conv2_bn')(x)
x = Activation('relu', name='block1_conv2_act')(x)

residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
residual = BatchNormalization()(residual)

x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
x = BatchNormalization(name='block2_sepconv1_bn')(x)
x = Activation('relu', name='block2_sepconv2_act')(x)
x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
x = BatchNormalization(name='block2_sepconv2_bn')(x)

x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)
x = layers.add([x, residual])

residual = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
residual = BatchNormalization()(residual)

x = Activation('relu', name='block3_sepconv1_act')(x)
x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
x = BatchNormalization(name='block3_sepconv1_bn')(x)
x = Activation('relu', name='block3_sepconv2_act')(x)
x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
x = BatchNormalization(name='block3_sepconv2_bn')(x)

x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(x)
x = layers.add([x, residual])

residual = Conv2D(728, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x) # Delete the parameter "strides=(2,2)" if you want to use small input images.
residual = BatchNormalization()(residual)

x = Activation('relu', name='block4_sepconv1_act')(x)
x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
x = BatchNormalization(name='block4_sepconv1_bn')(x)
x = Activation('relu', name='block4_sepconv2_act')(x)
x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
x = BatchNormalization(name='block4_sepconv2_bn')(x)

x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_pool')(x) # Comment this line if you want to use small input images. 
x = layers.add([x, residual])

"""
Middle Flow. Takes input of shape (b,b,728) and outputs same shape.
Here the middle flow is repeated 2 times. The original version repeats it 8 times.
In each block, the input of the block is added to the output before being passed to the next block, creating shortcuts. Therefore stacking the Middle Flow lots of time should not harm accuracy and convergence speed, but harms computation speed.
"""

for i in range(2):
	residual = x
	prefix = 'block' + str(i + 5)

	x = Activation('relu', name=prefix + '_sepconv1_act')(x)
	x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')(x)
	x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
	x = Activation('relu', name=prefix + '_sepconv2_act')(x)
	x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')(x)
	x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
	x = Activation('relu', name=prefix + '_sepconv3_act')(x)
	x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')(x)
	x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)

	x = layers.add([x, residual])

"""
Exit flow. Takes input of shape (c,c,728) and ouput tensors of activations of shape ~(c/2, c/2, 5270). Then, performs a spatial average over the first two dimensions of the tensors of shape (c,2/c,2,5270) to output vectors of shape (1,1,5270).
"""
residual = Conv2D(1536, (1, 1), strides=(2, 2),
              padding='same', use_bias=False)(x)
residual = BatchNormalization()(residual)

x = Activation('relu', name='block13_sepconv1_act')(x)
x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block13_sepconv1')(x)
x = BatchNormalization(name='block13_sepconv1_bn')(x)
x = Activation('relu', name='block13_sepconv2_act')(x)
x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False, name='block13_sepconv2')(x)
x = BatchNormalization(name='block13_sepconv2_bn')(x)

x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block13_pool')(x)
x = layers.add([x, residual])

x = SeparableConv2D(3072, (3, 3), padding='same', use_bias=False, name='block14_sepconv1')(x)
x = BatchNormalization(name='block14_sepconv1_bn')(x)
x = Activation('relu', name='block14_sepconv1_act')(x)

x = SeparableConv2D(5270, (3, 3), padding='same', use_bias=False, name='block14_sepconv2')(x)
x = BatchNormalization(name='block14_sepconv2_bn')(x)
x = Activation('relu', name='block14_sepconv2_act')(x)

x = GlobalAveragePooling2D(name='avg_pool')(x)

"""
Pass the 5270 averaged activations to a Softmax classifying layer with 5270 neurons.
"""
x = Dense(n_classes)(x) 
x = Activation('softmax')(x) 

cnn = Model(img_input, x)

cnn.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
cnn.summary()

# Prepare Image Generators

batch_size=32

train_folder = '/home/dlo/Documents/cdiscount/input/train'
validation_folder = '/home/dlo/Documents/cdiscount/input/validation'
test_folder = '/home/dlo/Documents/cdiscount/input/test'

train_datagen = ImageDataGenerator(
	rotation_range=8,
        rescale=1./255,
        zoom_range=0.1)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'input/train',  
        target_size=(img_height, img_width), 
        batch_size=batch_size)

test_generator = test_datagen.flow_from_directory(
        'input/validation',
        target_size=(img_height, img_width),
        batch_size=batch_size)

# Prepare checkpointer, will save weights after epochs for which the validation accuracy is better than the previous epoch. Prevents overfitting

checkpointer = ModelCheckpoint(filepath="weights_xceptionModified.hdf5", monitor='val_acc', verbose=1, save_best_only=True)


print("Training starts")
cnn.fit_generator(train_generator, steps_per_epoch=1500, epochs=310, verbose=1, validation_data=test_generator,validation_steps=400,callbacks=[checkpointer])
