# -*- coding: utf-8 -*-
#improvement #1, change target size/input shape from 128x128 down to 64x64 for faster computation
#improvement #2, add another convolution layer after the first one

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

#creating a ann with convolution
classifier = Sequential()

#classifier.add(Dense(units=6, init="glorot_uniform", activation="relu")) #3D array pixels for shape
#3x3 matrix with 32 feature maps in total, conventional. 3d array for colored img, RGB. 256 in term of intensity max/min
classifier.add(Convolution2D(32,3,3, input_shape=(64,64,3), activation="relu")) #tensorflow backend tuples = (128,128,3). Use relu to eliminate the neg values 

#maxpooling, normal pool_size is 2x2
classifier.add(MaxPooling2D(pool_size=(3,3))) #took out strides=1 for speed (no overlapping! remember the paper)


#second convo layer to look for better accuracy. we don't need input shape, keras arleady knows that.
classifier.add(Convolution2D(32,3,3, activation="relu"))
classifier.add(MaxPooling2D(pool_size=(3,3)))
#third convo layer with double feature filter size, 64 for better detection.

classifier.add(Convolution2D(64,3,3, activation="relu"))
classifier.add(MaxPooling2D(pool_size=(3,3)))

#Flattening to input the ann
classifier.add(Flatten()) #no argument needed

#Ann section with flattened stuff as input
#hidden layer, 68 nodes
classifier.add(Dense(68, activation="relu"))
#binary output uses sigmoid. Either a dog or a cat
classifier.add(Dense(1, activation="sigmoid"))

classifier.compile(optimizer="adam", loss="binary_crossentropy",metrics=["accuracy"])

#check out image processing in Keras documentation
train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True #for randomizing, not 2 same images in the 2 different batches
)

test_datagen = ImageDataGenerator(rescale=1./255) #data needs to be between 0 and 1
#target size is expected input of cnn, which is 128x128
training_data = train_datagen.flow_from_directory('dataset/training_set', target_size=(64,64), batch_size=32,class_mode="binary")
test_data = test_datagen.flow_from_directory('dataset/test_set', target_size=(64,64), batch_size=32, class_mode="binary")

#samples_per_epochs = number of training set size. Step per epoch usually is training size/batch size
classifier.fit_generator(training_data, steps_per_epoch=(int(8000/32)), epochs=20, validation_data=test_data, validation_steps=2000, use_multiprocessing=True)
