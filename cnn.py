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
classifier.fit_generator(training_data, steps_per_epoch=(int(8000/32)), epochs=25, validation_data=test_data, validation_steps=2000, use_multiprocessing=True)

#prediction
import numpy as np
from keras.preprocessing import image #to process image

picture1 = image.load_img("dataset/single_prediction/cat_or_dog_1.jpg", target_size=(64,64)) #same target size as the training set # of pixels

#turn image into 3d array, 64x64x3
picture1 = image.img_to_array(picture1)
picture1 = np.expand_dims(picture1,axis=0) #picture in first dimension/axis
#classifier.predict(picture1) #asking for 4 dimensions, error
result = classifier.predict(picture1) #only accept input in a batch with the ann/cnn, hence the expand_dims 

if result:
    print("dog")

else:
    print("cat")


picture2 = image.load_img("dataset/single_prediction/cat_or_dog_2.jpg", target_size = (64,64))
picture2 = image.img_to_array(picture2)
picture2 = np.expand_dims(picture2, axis=0)

result2 = classifier.predict(picture2)
if result2:
    print ('dog')
else:
    print("cat")

################################################################
#helper function to organize code later on

def result_output(result):
    if result:
        print("dog")
        return
    print("cat")
    return

def img_import(img): #img is the path
    picture = image.load_img(img, target_size=(64,64))
    picture = np.expand_dims(picture, axis=0)
    return picture


    













