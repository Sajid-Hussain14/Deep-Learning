# import tensorflow as tf
# from keras_preprocessing.image import ImageDataGenerator
# import numpy as np
#
# train_datagen = ImageDataGenerator(
#     rescale=1. / 255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True)
# training_set = train_datagen.flow_from_directory('Resources/Trainingset', target_size=(64, 64),
#                                                  batch_size=32, class_mode='categorical')
# test_datagen = ImageDataGenerator(rescale=1./255)
# test_set = test_datagen.flow_from_directory('Resources/Testset',target_size=(64,64),batch_size=32,class_mode='categorical')
# # from sklearn.preprocessing import LabelEncoder
# # lb = LabelEncoder()
# # training_set = lb.fit_transform(training_set)
#
# cnn = tf.keras.models.Sequential()
#
# cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu',padding='same',input_shape=[64,64,3]))
# cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,padding='same',strides=2))
#
# cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu',padding='same'))
# cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,padding='same',strides=2))
# cnn.add(tf.keras.layers.Dropout(0.5))
#
# cnn.add(tf.keras.layers.Flatten())
# cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))
#
# cnn.add(tf.keras.layers.Dense(units=7,activation='relu'))
# cnn.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
#
# cnn.fit(x=training_set,validation_data=test_set,epochs=20)

#####DATA AUGMENTATION#########3
from keras.preprocessing.image import ImageDataGenerator
import sklearn
import skimage
from skimage import io
import numpy as np

# Construct an instance of the ImageDataGenerator class
# Pass the augmentation parameters through the constructor.

datagen = ImageDataGenerator(
rotation_range=45, #Random rotation between 0 and 45
width_shift_range=0.2, #% shift
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.4,
horizontal_flip=True,
vertical_flip=True,
fill_mode='reflect', cval=125)

#Manually read each image and create an array to be supplied to datagen via flow method
dataset = []


from skimage import io
import os
from PIL import Image

image_directory = 'Resources/Trainingset/daisy/'
SIZE = 128
dataset = []

my_images = os.listdir(image_directory)
for i, image_name in enumerate(my_images):
    if (image_name.split('.')[1] == 'jpg'):
        image = io.imread(image_directory + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE,SIZE))
        dataset.append(np.array(image))

x = np.array(dataset)

i = 0
for batch in datagen.flow(x, batch_size=16,
                          save_to_dir='Resources/Trainingset/daisy',
                          save_prefix='ccc',
                          save_format='jpg'):
   i += 1
   if i > 20:
      break
