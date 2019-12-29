dir = 'C:/Users/Akshobhya/Documents/brain/yes'
j = 1
for filename in os.listdir(dir):
    j+=1
    a = filename
    filename = os.path.join(dir,filename)
    im = Image.open(filename)
    newsize = (224,224) 
    im = im.resize(newsize)
    name= 'C:/Users/Akshobhya/Documents/brain/newyes/'+ "Y"+ str(j)+".png" 
    im.save(name)
    
dir = 'C:/Users/Akshobhya/Documents/brain/no'
j = 1
for filename in os.listdir(dir):
    j+=1
    a = filename
    filename = os.path.join(dir,filename)
    im = Image.open(filename)
    newsize = (224,224) 
    im = im.resize(newsize)
    name= 'C:/Users/Akshobhya/Documents/brain/newno/' +"N"+str(j)+".png" 
    im.save(name)
    
    
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
j = 1

directory = 'Documents/brain/newno'

save_here = 'Documents/brain/newno2'
for filepath in os.listdir(directory):
    datagen = ImageDataGenerator(rotation_range=0.1, width_shift_range=0.01, 
    height_shift_range=0.01,shear_range=0.015, 
    zoom_range=0.01,channel_shift_range = 0.1, horizontal_flip=True,rescale= 1./255)
    filepath = os.path.join(directory, filepath)
    print(filepath)
    image_path = filepath
    image = np.expand_dims(plt.imread(image_path), 0)
    datagen.fit(image)
    for x, val in zip(datagen.flow(image, save_to_dir=save_here, save_prefix= str(j),save_format='png'),range(50)) :   
        pass
    
j = 1
directory = 'Documents/brain/newyes'

save_here = 'Documents/brain/newyes2'
for filepath in os.listdir(directory):
    datagen = ImageDataGenerator(rotation_range=0.1, width_shift_range=0.01, 
    height_shift_range=0.01,shear_range=0.015, 
    zoom_range=0.01,channel_shift_range = 0.1, horizontal_flip=True,rescale= 1./255)
    filepath = os.path.join(directory, filepath)
    print(filepath)
    image_path = filepath
    image = np.expand_dims(plt.imread(image_path), 0)
    datagen.fit(image)
    for x, val in zip(datagen.flow(image, save_to_dir=save_here, save_prefix= str(j),save_format='jpg'),range(50)) :   
        pass
