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
    grab_random_samples_from_dir
)

# Kill Tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Dictionary
dict_ = {
    '0' : "Cat",
    '1' : "Dog"
}

'''
Preprocess image before prediction
'''

# Load model into memory
cnn_model = load_model('models/my_model_1600_400.h5')

# Read prediction image
image = read_image_and_set_shape("train/cat_samples/cat.208.jpg")

# # Resize image
resized_image = resize_image_with_new_shape(image, [64,64])

#Greyscale image
greyscale_image = convert_to_greyscale(resized_image)

x = tf.Variable(greyscale_image, name='x')
model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    result = session.run(x)

# Reexpand shape from (64, 64) to (64, 64, 3)
result = np.stack((result,) * 3, axis=2)

# Predictions require an input of 4 dimenions. In other words, an array of image arrays
results = np.array([result])


'''
Make predictions
'''

prediction = cnn_model.predict(results).tolist()[0][0]

prediction_key = dict_[str(int(prediction))]

print("Prediction: ", prediction_key)

del cnn_model

'''
Show prediction image, just uncomment this to view

'''

# Squeeze image to plot
result = np.squeeze(result)
imgplot = plt.imshow(result)
imgplot.set_cmap('gray')

plt.show(imgplot)
