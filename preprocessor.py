import os 
from numpy import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
import shutil
import scipy.misc
import time
from methods import(
    grab_random_samples_from_dir,
    read_image_and_set_shape,
    convert_to_greyscale,
    resize_image_with_new_shape
)

@timer
def main(params):
    
    '''
    Preprocess
    '''
    
    random_samples = grab_random_samples_from_dir(params['image_folder'], int(params['number_of_random_samples']))
    
    list_of_originals = []
    list_of_modified = []
    
    for image in random_samples:
        # Read image 
        original_image = read_image_and_set_shape(params['image_folder'] + "/"+image)

        # Resize image
        resized_image = resize_image_with_new_shape(original_image, [64,64])
        
        # Grey scale image
        greyscale_image = convert_to_greyscale(resized_image)

        # Append images to list
        
        var = tf.Variable(greyscale_image, name='{}'.format("x"))
        
        list_of_originals.append(original_image)
        list_of_modified.append(var)
    
    # Create Tensorflow Variable 
    tf_original_list = [tf.Variable(image, name="tf_original_list") for image in list_of_originals]
    tf_modified_list = [tf.Variable(image, name="tf_modified_list") for image in list_of_modified]
    
    model = tf.global_variables_initializer()

    original_images_results = []
    modified_images_results = []
        
    with tf.Session() as session:
        session.run(model)

        for image_var in tf_original_list:            
            original_images_results.append(session.run(image_var))
                
        for image_var in tf_modified_list:
            modified_images_results.append(session.run(image_var))

    '''
    Save images
    '''
        
#     # Make new folder
    if os.path.exists(params['name_of_save_dir']):
        shutil.rmtree(params['name_of_save_dir'])
    os.mkdir(params['name_of_save_dir'])
    
    for i in range(0, len(modified_images_results)):
        scipy.misc.imsave(params['name_of_save_dir'] + "/" + params['name_of_greyscale_files'] + "{}.jpg".format(i + 1), modified_images_results[i])
    
    print("\nCompleted rescaling and converting images to grayscale! ...")


if __name__ == '__main__':  
    
    params = {}    
    try: 
        params['image_folder'] = os.getcwd() + '/' + input("Enter a folder to grab random_images from: ")
        params['number_of_random_samples'] = input("Enter a number of random samples: ")
        params['name_of_save_dir'] = input("Enter the name of a file to save to (optional): ")
        params['name_of_greyscale_files'] = input("Enter the name of a greyscaled_file (optional): ")
        params['rescale_size'] = input("Enter a rescale factor (0 < x <= 1). Default is 0.5 : ")
    except Exception as e:
        print(e)
    
    if len(params['name_of_save_dir']) == 0: 
        params['name_of_save_dir'] = "greyscale_images"
    
    if len(params['name_of_greyscale_files']) == 0: 
        params['name_of_greyscale_files'] = "grey_image"
    
    if len(params['rescale_size']) == 0: 
        params['rescale_size'] = 0.5
                
    '''
    EXAMPLE of inputs

    #     params['image_folder'] = 'train/dog_samples'
    #     params['number_of_random_samples'] = "3"
    #     params['name_of_save_dir'] = "grey_dog_images"
    #     params['name_of_greyscale_files'] = "grey_dogs"
    #     params['rescale_size'] = "0.4"

    '''
            
    main(params)