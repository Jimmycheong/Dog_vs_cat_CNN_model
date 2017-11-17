'''
The purpose of this script is to train a CNN model. 

The model is behind on using the keras library and consists of the following layers:
- Convolutional Layer 1
- Pooling Layer 1
- Convolutional Layer 2
- Pooling Layer 2
- Flattening Layer
- Hidden Layer  
- Outputer Layer

'''

import time
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator

# Start timing
time_0 = time.time()
    

'''
Build layers of CNN model

'''
classifier = Sequential()

# Convolve layer 1 
classifier.add(Convolution2D(32, (3,3), input_shape=(64,64,3), activation="relu"))
# Pool layer 1 
classifier.add(MaxPooling2D(pool_size= (2,2)))

# Convolve layer 2 
classifier.add(Convolution2D(64, (3,3), activation="relu"))

# Pool layer 2 
classifier.add(MaxPooling2D(pool_size= (2,2)))

classifier.add(Flatten())

classifier.add(Dense(activation="relu", units = 128))
classifier.add(Dense(activation="sigmoid", units= 1))

classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

'''
Fitting the CNN to the images
'''

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'grey_images/training',
    target_size=(64,64),
    batch_size=32,
    class_mode="binary"
)

test_set = test_datagen.flow_from_directory(
    'grey_images/testing',
    target_size=(64,64),
    batch_size=32,
    class_mode="binary"
)

classifier.fit_generator(
    training_set,
    steps_per_epoch=20,
    nb_epoch=5,
    validation_data=test_set,
    validation_steps=5
)


print("Time elapsed: ", time.time() - time_0)

'''
Save the model
'''


model_file_name = "models/my_model.h5"
print("\nSaving model as {} ...".format(model_file_name))
classifier.save(model_file_name)
