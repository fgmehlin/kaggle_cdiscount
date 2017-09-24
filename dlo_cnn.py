import os
import glob
from scipy import misc
import numpy as np
import keras.models as models
from keras import layers
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Activation, Dropout, Dense, BatchNormalization, Input, GlobalAveragePooling2D, AveragePooling2D, Conv2D, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm_notebook as tqdmn
from keras.callbacks import ModelCheckpoint

""" 
Implemented several modules of Google's InceptionV4 from https://arxiv.org/pdf/1602.07261.pdf .
Figure 9 of the pdf shows the high level architecture of the network, how modules follow each other. Figures 3 to 8 detail the modules.

In this script following modules are implemented:
Stem, Inception-A, Inception-B, Inception-C, Reduction-B.
Average Pooling, Dropout, Softmax are built in Keras. 

General guideline on how to use the modules/functions of InceptionV4 (see figure 9 from above pdf for more details):
	- The input data has to be first processed by the Stem module
	- The output of the Stem module has to be processed by an Inception module. 
	- Inception modules do not change the spatial shape of the tensors. Inception modules of the same type can be stacked one after another. To pass from an Inception module to an Inception module of another type, a Reduction function should be used in between.
	- Reduction modules can be used to spatially reduce the spatial shape of a tensor, between two Inception modules. Reduction modules reduce the spatial shape, but augment the depth (a,a,b) -> 	(c,c,d) with c<a and d>b.
	

In general, Inception modules "do the learning work" and Reduction modules reduce the spatial shape. Inception modules of the same type can be stacked after each other.

For cdiscount's dataset, the original shape is (180,180). 
	-If you don't reshape it, pass it through the Stem module to give it shape ~ (17, 17, 384). You can then stack Inception modules, but you should use the Reduction-B module at some point between inception modules to reduce spatially from (17, 17, .) to (8, 8, .).
	-If you reshape it to something smaller, like (80,80) or smaller, pass it to the Stem module to give it a shape ~ (7, 7, 384). You can then stack Inception modules before the output, but you should not need Reduction modules.

I also tested it on MNIST using Input -> Stem(modified a bit: all padding to 'same' instead of some having 'valid') -> Inception-C -> GlobalAveragePooling2D -> Dropout(0.4) -> SoftMax(10):

img_input = Input(shape=(28,28,1))
x = stem(img_input)
x = inceptBig(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(10)(x)
x = Activation('softmax')(x)


#### HOW TO MODIFY THE NETWORK ####

-In general, modules apply changes to the depth, and/or witdh,heigth of data. 
-Inception modules SHALL NOT reduce spatially. Use only stride=(1,1), padding of type 'same', and no Pooling. Use Reduction modules to reduce spatially.
-You can make Reduction modules reduce further spatially by applying larger strides, e.g. strides=(3,3) instead of (2,2).
-You can make the Stem module reduce a bit less spatially by changing all the padding type to 'same'.
-You can make the overall network less expensive by reducing the depth of the output of the modules. To do that, use the fagFactor passed as argument to the modules. You should use a fagFactor between 0 and 0.8, where the bigger the fagFactor, the smaller the network. A fagFactor of 0 will use the default dimension of Google's InceptionV4


"""

def stem(img_input, fagFactor) : # Stem module, processes input of shape (a, a, 3) and ouputs tensor of shape ~ (a/10, a/10, 384), to be passed to an Inception module

	
	if fagFactor < 0 :
		raise ValueError('fagFactor cannot be negative')
	if fagFactor > 0.8 :
		raise ValueError('fagFactor cannot be larger than 0.8')

	fagFactor = 1.0-fagFactor
	x = Conv2D(int(fagFactor*32), (3, 3), strides=(2,2), padding='valid')(img_input)
	x = BatchNormalization(axis=3,scale=False)(x)
	x = Activation('relu')(x)

	x = Conv2D(int(fagFactor*32), (3, 3), strides=(1,1), padding='valid')(x)
	x = BatchNormalization(axis=3,scale=False)(x)
	x = Activation('relu')(x)

	x = Conv2D(int(fagFactor*64), (3, 3), strides=(1,1), padding='valid')(x)
	x = BatchNormalization(axis=3,scale=False)(x)
	x = Activation('relu')(x)

	branch1 = MaxPool2D((3, 3), strides=(2, 2), padding='valid')(x)

	branch2 = Conv2D(int(fagFactor*96), (3, 3), strides=(2, 2), padding='valid')(x)
	branch2 = BatchNormalization(axis=3,scale=False)(branch2)
	branch2 = Activation('relu')(branch2)

	x = layers.concatenate([branch1, branch2], axis=3)

	branch1 = Conv2D(int(fagFactor*64), (1, 1), strides=(1,1), padding='same')(x)
	branch1 = BatchNormalization(axis=3,scale=False)(branch1)
	branch1 = Activation('relu')(branch1)

	branch1 = Conv2D(int(fagFactor*96), (3, 3), strides=(1,1), padding='valid')(branch1)
	branch1 = BatchNormalization(axis=3,scale=False)(branch1)
	branch1 = Activation('relu')(branch1)

	branch2 = Conv2D(int(fagFactor*64), (1, 1), strides=(1,1), padding='same')(x)
	branch2 = BatchNormalization(axis=3,scale=False)(branch2)
	branch2 = Activation('relu')(branch2)

	branch2 = Conv2D(int(fagFactor*64), (7, 1), strides=(1,1), padding='same')(branch2)
	branch2 = BatchNormalization(axis=3,scale=False)(branch2)
	branch2 = Activation('relu')(branch2)

	branch2 = Conv2D(int(fagFactor*64), (1, 7), strides=(1,1), padding='same')(branch2)
	branch2 = BatchNormalization(axis=3,scale=False)(branch2)
	branch2 = Activation('relu')(branch2)

	branch2 = Conv2D(int(fagFactor*96), (3, 3), strides=(1,1), padding='valid')(branch2)
	branch2 = BatchNormalization(axis=3,scale=False)(branch2)
	branch2 = Activation('relu')(branch2)

	x = layers.concatenate([branch1, branch2], axis=3)

	branch1 = MaxPool2D(strides=(2, 2), padding='same')(x)

	branch2 = Conv2D(int(fagFactor*128), (1, 1), strides=(2,2), padding='same')(x)
	branch2 = BatchNormalization(axis=3,scale=False)(branch2)
	branch2 = Activation('relu')(branch2)

	x = layers.concatenate([branch1, branch2], axis=3)
	return x


def inceptBig(x, fagFactor) : # Inception-A module that takes input of size (a, a, b) and outputs (a, a, 384). a should be big, ~ 35
	
	if fagFactor < 0 :
		raise ValueError('fagFactor cannot be negative')
	if fagFactor > 0.8 :
		raise ValueError('fagFactor cannot be larger than 0.8')

	fagFactor = 1.0-fagFactor
	branch1 = AveragePooling2D(strides=(1,1), padding='same')(x)

	branch1 = Conv2D(int(fagFactor*96), (1, 1), strides=(1,1), padding='same')(branch1)
	branch1 = BatchNormalization(axis=3,scale=False)(branch1)
	branch1 = Activation('relu')(branch1)

	branch2 = Conv2D(int(fagFactor*96), (1, 1), strides=(1,1), padding='same')(x)
	branch2 = BatchNormalization(axis=3,scale=False)(branch2)
	branch2 = Activation('relu')(branch2)

	branch3 = Conv2D(int(fagFactor*64), (1, 1), strides=(1,1), padding='same')(x)
	branch3 = BatchNormalization(axis=3,scale=False)(branch3)
	branch3 = Activation('relu')(branch3)

	branch3 = Conv2D(int(fagFactor*96), (3, 3), strides=(1,1), padding='same')(branch3)
	branch3 = BatchNormalization(axis=3,scale=False)(branch3)
	branch3 = Activation('relu')(branch3)

	branch4 = Conv2D(int(fagFactor*64), (1, 1), strides=(1,1), padding='same')(x)
	branch4 = BatchNormalization(axis=3,scale=False)(branch4)
	branch4 = Activation('relu')(branch4)

	branch4 = Conv2D(int(fagFactor*96), (3, 3), strides=(1,1), padding='same')(branch4)
	branch4 = BatchNormalization(axis=3,scale=False)(branch4)
	branch4 = Activation('relu')(branch4)

	branch4 = Conv2D(int(fagFactor*96), (3, 3), strides=(1,1), padding='same')(branch4)
	branch4 = BatchNormalization(axis=3,scale=False)(branch4)
	branch4 = Activation('relu')(branch4)

	x = layers.concatenate([branch1, branch2, branch3, branch4], axis=3)
	return x


def inceptMed(x, fagFactor) : # Inception-B module that takes input of size (a, a, b) and outputs (a, a, 1024). a should be medium, ~ 17
	
	if fagFactor < 0 :
		raise ValueError('fagFactor cannot be negative')
	if fagFactor > 0.8 :
		raise ValueError('fagFactor cannot be larger than 0.8')

	fagFactor = 1.0-fagFactor	
	branch1 = AveragePooling2D(strides=(1,1), padding='same')(x)

	branch1 = Conv2D(int(fagFactor*128), (1, 1), strides=(1,1), padding='same')(branch1)
	branch1 = BatchNormalization(axis=3,scale=False)(branch1)
	branch1 = Activation('relu')(branch1)

	branch2 = Conv2D(int(fagFactor*384), (1, 1), strides=(1,1), padding='same')(x)
	branch2 = BatchNormalization(axis=3,scale=False)(branch2)
	branch2 = Activation('relu')(branch2)

	branch3 = Conv2D(int(fagFactor*192), (1, 1), padding='same')(x)
	branch3 = BatchNormalization(axis=3,scale=False)(branch3)
	branch3 = Activation('relu')(branch3)

	branch3 = Conv2D(int(fagFactor*224), (1, 7), padding='same')(branch3)
	branch3 = BatchNormalization(axis=3,scale=False)(branch3)
	branch3 = Activation('relu')(branch3)

	branch3 = Conv2D(int(fagFactor*256), (1, 7), padding='same')(branch3)
	branch3 = BatchNormalization(axis=3,scale=False)(branch3)
	branch3 = Activation('relu')(branch3)

	branch4 = Conv2D(int(fagFactor*192), (1, 1), padding='same')(x)
	branch4 = BatchNormalization(axis=3,scale=False)(branch4)
	branch4 = Activation('relu')(branch4)

	branch4 = Conv2D(int(fagFactor*192), (1, 7), padding='same')(branch4)
	branch4 = BatchNormalization(axis=3,scale=False)(branch4)
	branch4 = Activation('relu')(branch4)

	branch4 = Conv2D(int(fagFactor*224), (7, 1), padding='same')(branch4)
	branch4 = BatchNormalization(axis=3,scale=False)(branch4)
	branch4 = Activation('relu')(branch4)

	branch4 = Conv2D(int(fagFactor*224), (1, 7), padding='same')(branch4)
	branch4 = BatchNormalization(axis=3,scale=False)(branch4)
	branch4 = Activation('relu')(branch4)

	branch4 = Conv2D(int(fagFactor*256), (7, 1), padding='same')(branch4)
	branch4 = BatchNormalization(axis=3,scale=False)(branch4)
	branch4 = Activation('relu')(branch4)

	x = layers.concatenate([branch1, branch2, branch3, branch4], axis=3)

	return x
	

def inceptSmall(x, fagFactor) : # Inception-C that takes input of size (a, a, b) and outputs (a, a, 1536). a should be small, ~ 8
	
	if fagFactor < 0 :
		raise ValueError('fagFactor cannot be negative')
	if fagFactor > 0.8 :
		raise ValueError('fagFactor cannot be larger than 0.8')

	fagFactor = 1.0-fagFactor
	branch1 = AveragePooling2D(strides=(1,1), padding='same')(x)

	branch1 = Conv2D(int(fagFactor*256), (1, 1), strides=(1,1), padding='same')(branch1)
	branch1 = BatchNormalization(axis=3,scale=False)(branch1)
	branch1 = Activation('relu')(branch1)

	branch2 = Conv2D(int(fagFactor*256), (1, 1), strides=(1,1), padding='same')(x)
	branch2 = BatchNormalization(axis=3,scale=False)(branch2)
	branch2 = Activation('relu')(branch2)

	branch3 = Conv2D(int(fagFactor*384), (1, 1), padding='same')(x)
	branch3 = BatchNormalization(axis=3,scale=False)(branch3)
	branch3 = Activation('relu')(branch3)

	branch3a = Conv2D(int(fagFactor*256), (3, 1), padding='same')(branch3)
	branch3a = BatchNormalization(axis=3,scale=False)(branch3a)
	branch3a = Activation('relu')(branch3a)

	branch3b = Conv2D(int(fagFactor*256), (1, 3), padding='same')(branch3)
	branch3b = BatchNormalization(axis=3,scale=False)(branch3b)
	branch3b = Activation('relu')(branch3b)

	branch4 = Conv2D(int(fagFactor*384), (1, 1), padding='same')(x)
	branch4 = BatchNormalization(axis=3,scale=False)(branch4)
	branch4 = Activation('relu')(branch4)

	branch4 = Conv2D(int(fagFactor*448), (3, 1), padding='same')(branch4)
	branch4 = BatchNormalization(axis=3,scale=False)(branch4)
	branch4 = Activation('relu')(branch4)

	branch4 = Conv2D(int(fagFactor*512), (1, 3), padding='same')(branch4)
	branch4 = BatchNormalization(axis=3,scale=False)(branch4)
	branch4 = Activation('relu')(branch4)

	branch4a = Conv2D(int(fagFactor*256), (3, 1), padding='same')(branch4)
	branch4a = BatchNormalization(axis=3,scale=False)(branch4a)
	branch4a = Activation('relu')(branch4a)

	branch4b = Conv2D(int(fagFactor*256), (1, 3), padding='same')(branch4)
	branch4b = BatchNormalization(axis=3,scale=False)(branch4b)
	branch4b = Activation('relu')(branch4b)

	x = layers.concatenate([branch1, branch2, branch3a, branch3b, branch4a, branch4b], axis=3)

	return x


def reduceMedToSmall(x, fagFactor) : # Reduction-B module that takes input of size ~ (a, a, b) and outputs (c, c, ~1536) with c ~= a/2. a should be ~=7
	
	if fagFactor < 0 :
		raise ValueError('fagFactor cannot be negative')
	if fagFactor > 0.9 :
		raise ValueError('fagFactor cannot be larger than 0.8')
	fagFactor = 1.0-fagFactor
	branch1 = MaxPool2D((3,3), strides=(2,2), padding = 'valid')(x)

	branch2 = Conv2D(int(fagFactor*192), (1, 1), padding='same')(x)
	branch2 = BatchNormalization(axis=3,scale=False)(branch2)
	branch2 = Activation('relu')(branch2)

	branch2 = Conv2D(int(fagFactor*192), (3, 3), strides=(2,2), padding='valid')(branch2)
	branch2 = BatchNormalization(axis=3,scale=False)(branch2)
	branch2 = Activation('relu')(branch2)

	branch3 = Conv2D(int(fagFactor*256), (1, 1), padding='same')(x)
	branch3 = BatchNormalization(axis=3,scale=False)(branch3)
	branch3 = Activation('relu')(branch3)

	branch3 = Conv2D(int(fagFactor*256), (1, 7), padding='same')(branch3)
	branch3 = BatchNormalization(axis=3,scale=False)(branch3)
	branch3 = Activation('relu')(branch3)

	branch3 = Conv2D(int(fagFactor*320), (7, 1), padding='same')(branch3)
	branch3 = BatchNormalization(axis=3,scale=False)(branch3)
	branch3 = Activation('relu')(branch3)

	branch3 = Conv2D(int(fagFactor*320), (3, 3), strides=(2,2), padding='valid')(branch3)
	branch3 = BatchNormalization(axis=3,scale=False)(branch3)
	branch3 = Activation('relu')(branch3)

	x = layers.concatenate([branch1, branch2, branch3], axis=3)
	return x


# Data

train_folder = '/home/dlo/Documents/cdiscount/input/train'
validation_folder = '/home/dlo/Documents/cdiscount/input/validation'
test_folder = '/home/dlo/Documents/cdiscount/input/test'

img_width = 80 # Change to what you want
img_height = 80 # Change to what you want
n_channel = 3

class_list = os.listdir(train_folder)
n_classes = len(class_list)
n_train_examples = 12371293 # Change to what you want
n_test_examples = 3095080 # Change to what you want

# Prepare image generator

# Build network

fagFactor = 0
img_input = Input(shape=(img_width,img_height,n_channel)) # Here I have set img_width and img_height to 80. -> img_input is the Input layer with shape (80,80,3)
x = stem(img_input, fagFactor) # apply the Stem module on the Input layer and return x of shape (7,7,384)
x = inceptSmall(x,fagFactor) # apply the Inception-C module on x, and returns x with shape (7,7,1536)
x = inceptSmall(x,fagFactor) # apply the Inception-C module on x, and returns x with shape (7,7,1536)
x = GlobalAveragePooling2D()(x) # Average pool spatially on x, returns x with shape (1536)
x = Dropout(0.2)(x) # Drops 20% of above activations during training
x = Dense(n_classes)(x) 
x = Activation('softmax')(x) 

cnn = Model(img_input, x)
cnn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
cnn.summary()

batch_size=32

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

checkpointer = ModelCheckpoint(filepath="weights1.hdf5", monitor='val_acc', verbose=1, save_best_only=True)


print("Training starts")
cnn.fit_generator(train_generator, steps_per_epoch=300, epochs=100, verbose=1, validation_data=test_generator,validation_steps=100,callbacks=[checkpointer])
