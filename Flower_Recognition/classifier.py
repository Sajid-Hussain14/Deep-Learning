from main import load_data

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

(feature, labels) = load_data()
x_train, x_test, y_train, y_test = train_test_split(feature, labels, test_size=0.1)
categories = ['caesal','daisy','eupho','lantana','llily','testdata']

input_layer = tf.keras.layers.Input([224,224,3]) #shape of input feature
#passing input_layer to other layers

con1 = tf.keras.layers.Conv2D(filters =32, kernel_size = (5,5), padding = 'Same', activation='relu')(input_layer)

pool1 = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(con1)
con2 = tf.keras.layers.Conv2D(filters =64, kernel_size = (3,3), padding='Same', activation='relu')(pool1)
pool2 = tf.keras.layers.MaxPooling2D(pool_size =(2,2), strides=(2,2))(con2)
con3 = tf.keras.layers.Conv2D(filters =96, kernel_size =(3,3), padding='Same', activation='relu')(pool2)
pool3 = tf.keras.layers.MaxPooling2D(pool_size =(2,2), strides=(2,2))(con3)
con4 = tf.keras.layers.Conv2D(filters =96, kernel_size =(3,3), padding='Same', activation='relu')(pool3)
pool4 = tf.keras.layers.MaxPooling2D(pool_size =(2,2), strides=(2,2))(con4)

flt1 = tf.keras.layers.Flatten()(pool4)

dn1 = tf.keras.layers.Dense(510, activation='relu')(flt1)
out = tf.keras.layers.Dense(5,activation='softmax')(dn1)
                            #5 chiii neurons

model = tf.keras.Model(input_layer, out)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit( x_train, y_train, batch_size=100, epochs=20)
model.save('mymodel.h5')

