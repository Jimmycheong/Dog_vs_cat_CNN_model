'''
The purprose of this script is to make predictions for a photo
'''

import os
import scipy
import tensorflow as tf
from keras.models import load_model
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from methods import (
    read_image_and_set_shape,
    convert_to_greyscale,
    resize_image_with_new_shape,
    grab_random_samples_from_dir,
    FILLER
)

# Kill Tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

'''
SETUP
'''
dict_ = {
    '0' : "Cat",
    '1' : "Dog"
}

RANDOM_COUNT = 8

LOAD_DIR = "models/my_model_1600_400.h5"
DOG_DIR = "train/dogs/"
CAT_DIR = "train/cats/"


'''
Load model into memory
'''
cnn_model = load_model(LOAD_DIR)

'''
Grab random images 
'''
random_dogs = grab_random_samples_from_dir(DOG_DIR, RANDOM_COUNT).tolist()
random_dog_loc = [DOG_DIR + x for x in random_dogs]

random_cats= grab_random_samples_from_dir(CAT_DIR, RANDOM_COUNT).tolist()
random_cats_loc = [CAT_DIR + x for x in random_cats]

images = random_dog_loc + random_cats_loc

# Generate test labels (e.g. y_test)
y_test = ["Dog" if "dogs" in name else "Cat" for name in images]

operations = []

'''
Convert images by resizing and greyscaling
'''

# Read prediction image
for i in images: 
    image = read_image_and_set_shape(i)
    resized_image = resize_image_with_new_shape(image, [64,64])
    greyscale_image = convert_to_greyscale(resized_image)
    operations.append(tf.Variable(greyscale_image, name='{}'.format(i)))
    
model = tf.global_variables_initializer()

results = []

with tf.Session() as session:
    session.run(model)
    
    for i in range(0, len(operations)):
        result = session.run(operations[i])
        result = np.stack((result,) * 3, axis=2) # Re-expand shape from (64, 64) to (64, 64, 3)
        # Predictions require an input of 4 dimenions. In other words, an array of image arrays
        results.append(result)

        
results = np.array(results)

'''
Make predictions
'''    

pred = cnn_model.predict(results).tolist()
discrete_pred = [dict_["1"] if x[0] > 0.5 else dict_["0"] for x in pred]

from sklearn.metrics import accuracy_score
acc = accuracy_score(discrete_pred, y_test)

print("RESULTS:\n{}\nAccuracy of model:{:5.4f}\n{}".format(FILLER, acc, FILLER))

del cnn_model

'''
**VISUALISATION**

Show prediction image, just uncomment this to view

'''

fig, axarr = plt.subplots(len(results),1)
fig.set_figheight(100, forward=True)
fig.set_figwidth(100, forward=True)

for i in range(0, len(results)):
    result = pred[i][0]
    label = dict_["1" if result > 0.5 else "0"]
    confidence = result if result > 0.5 else (1 - result)
    axarr[i].set_title(
        "Prediction: {}\nConfidence: {:5.3f}".format(label, confidence))
    grayplot = axarr[i].imshow(results[i])
    grayplot.set_cmap('gray')

plt.tight_layout()
plt.axis('scaled')
plt.show()
'''
The purprose of this script is to make predictions for a photo
'''

import os
import scipy
import tensorflow as tf
from keras.models import load_model
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from methods import (
    read_image_and_set_shape,
    convert_to_greyscale,
    resize_image_with_new_shape,
    grab_random_samples_from_dir,
    FILLER
)

# Kill Tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

'''
SETUP
'''
dict_ = {
    '0' : "Cat",
    '1' : "Dog"
}

RANDOM_COUNT = 8

LOAD_DIR = "../models/my_model_1600_400.h5"
DOG_DIR = "../train/dogs/"
CAT_DIR = "../train/cats/"


'''
Load model into memory
'''
cnn_model = load_model(LOAD_DIR)

'''
Grab random images 
'''
random_dogs = grab_random_samples_from_dir(DOG_DIR, RANDOM_COUNT).tolist()
random_dog_loc = [DOG_DIR + x for x in random_dogs]

random_cats= grab_random_samples_from_dir(CAT_DIR, RANDOM_COUNT).tolist()
random_cats_loc = [CAT_DIR + x for x in random_cats]

images = random_dog_loc + random_cats_loc

# Generate test labels (e.g. y_test)
y_test = ["Dog" if "dogs" in name else "Cat" for name in images]

operations = []

'''
Convert images by resizing and greyscaling
'''

# Read prediction image
for i in images: 
    image = read_image_and_set_shape(i)
    resized_image = resize_image_with_new_shape(image, [64,64])
    greyscale_image = convert_to_greyscale(resized_image)
    operations.append(tf.Variable(greyscale_image, name='{}'.format(i)))
    
model = tf.global_variables_initializer()

results = []

with tf.Session() as session:
    session.run(model)
    
    for i in range(0, len(operations)):
        result = session.run(operations[i])
        result = np.stack((result,) * 3, axis=2) # Re-expand shape from (64, 64) to (64, 64, 3)
        # Predictions require an input of 4 dimenions. In other words, an array of image arrays
        results.append(result)

        
results = np.array(results)

'''
Make predictions
'''    

pred = cnn_model.predict(results).tolist()
discrete_pred = [dict_["1"] if x[0] > 0.5 else dict_["0"] for x in pred]

from sklearn.metrics import accuracy_score
acc = accuracy_score(discrete_pred, y_test)

print("RESULTS:\n{}\nAccuracy of model:{:5.4f}\n{}".format(FILLER, acc, FILLER))

del cnn_model

'''
**VISUALISATION**

Show prediction image, just uncomment this to view

'''

fig, axarr = plt.subplots(len(results),1)
fig.set_figheight(100, forward=True)
fig.set_figwidth(100, forward=True)

for i in range(0, len(results)):
    result = pred[i][0]
    label = dict_["1" if result > 0.5 else "0"]
    confidence = result if result > 0.5 else (1 - result)
    axarr[i].set_title(
        "Prediction: {}\nConfidence: {:5.3f}".format(label, confidence))
    grayplot = axarr[i].imshow(results[i])
    grayplot.set_cmap('gray')

plt.tight_layout()
plt.axis('scaled')
plt.show()
