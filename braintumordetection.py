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
    for x, val in zip(datagen.flow(image, save_to_dir=save_here, save_prefix= str(j),save_format='png'),range(100)) :   
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
    for x, val in zip(datagen.flow(image, save_to_dir=save_here, save_prefix= str(j),save_format='jpg'),range(100)) :   
        pass

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

DATADIR = r"C:\Users\Akshobhya\Documents\ML datasets\brain-mri-images-for-brain-tumor-detection"

CATEGORIES = ["newno3", "newyes3"]
data = []

def create_data():
    for category in CATEGORIES:  

        path = os.path.join(DATADIR,category) 
        class_num = CATEGORIES.index(category)  

        for img in tqdm(os.listdir(path)):
            img_array = cv2.imread(os.path.join(path,img))  
            data.append([img_array, class_num]) 
            
    
create_data()
import random
random.shuffle(data)
for i in range(30):
    print(data[i])
X_train  = []
y_train =  []
X_test = []
y_test = []


for i in range(len(data)):
        if(i<=len(data)*0.9):
            a,b = data[i]
            X_train.append(a)
            y_train.append(b)
        else:
            a,b = data[i]
            X_test.append(a)
            y_test.append(b)
            
            
 # load base model
IMG_SIZE = (224,224)
vgg16_weight_path = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
base_model = VGG16(
    weights=vgg16_weight_path,
    include_top=True, 
    input_shape=(224,224,3)
)
NUM_CLASSES = 1

model = Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(NUM_CLASSES, activation='sigmoid'))

model.layers[0].trainable = False

model.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(lr=1e-4),
    metrics=['accuracy']
)

model.summary()
EPOCHS = 30
es = EarlyStopping(
    monitor='val_acc', 
    mode='max',
    patience=6
)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=50,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=25,
    callbacks=[es]
)
# validate on val set
predictions = model.predict(X_val_prep)
predictions = [1 if x>0.5 else 0 for x in predictions]

accuracy = accuracy_score(y_val, predictions)
print('Val Accuracy = %.2f' % accuracy)

