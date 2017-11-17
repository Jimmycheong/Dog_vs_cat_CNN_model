'''
The following file parses the training data into dog and cat images

'''

import os 

photo_directory = "train"

for file in os.listdir(photo_directory): 
    if "cat" in file and "jpg" in file:
        os.rename(photo_directory + '/'+ file, "{}/cats/{}".format(photo_directory, file))
    elif "dog" in file and "jpg" in file: 
        os.rename(photo_directory + '/'+ file, "{}/dogs/{}".format(photo_directory, file))
    else:
        pass