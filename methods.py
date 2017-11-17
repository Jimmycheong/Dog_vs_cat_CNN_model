import time
import tensorflow as tf
import os
from numpy import random
import matplotlib.image as mpimg

FILLER = "*" * 20

def timer(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('Operation took {}s'.format((time2-time1)))
        return ret
    return wrap


def grab_random_samples_from_dir(dir, number_of_random_samples):
    samples = os.listdir(dir)
    return random.choice(samples, number_of_random_samples)

def read_image_and_set_shape(image_location):
    image = mpimg.imread(image_location)
    file_contents = tf.read_file(image_location)
    decoded_image = tf.image.decode_png(file_contents, dtype=tf.uint8, channels=3)   
    decoded_image.set_shape(image.shape)

    return decoded_image

def convert_to_greyscale(image):
    rgb_image_float = tf.image.convert_image_dtype(image, tf.float32)
    grayscale_image = tf.image.rgb_to_grayscale(rgb_image_float)
    grayscale_image = tf.squeeze(grayscale_image)

    return grayscale_image


def resize_image_with_scale_factor(decoded_image, scale_factor):
    rescaled_row = int(decoded_image.shape.as_list()[0] * scale_factor)
    rescaled_col = int(decoded_image.shape.as_list()[1] * scale_factor)

    resize_shape = tf.stack([rescaled_row,rescaled_col])
    resized_image = tf.image.resize_images(
        decoded_image, 
        resize_shape,
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    resized_image = tf.squeeze(resized_image)
    
    return resized_image

def resize_image_with_new_shape(decoded_image, shape):

    resize_shape = tf.stack(shape)
    resized_image = tf.image.resize_images(
        decoded_image, 
        resize_shape,
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    resized_image = tf.squeeze(resized_image)
    
    return resized_image