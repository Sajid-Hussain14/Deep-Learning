import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf

#we can use Conolutional NN but it is beginner task
#loading from tensorflow training data testing data
# mnist = tf.keras.datasets.mnist
# (x_train,y_train), (x_test,y_test) = mnist.load_data()

# #now normalizing(pixels not digits as it will be easy for calculations) 0 - 255
# #basically scaling down
#
# x_train = tf.keras.utils.normalize(x_train, axis = 1)
# x_test = tf.keras.utils.normalize(x_test, axis = 1)
#
# #WORKING NOW ON NN
#
# model = tf.keras.models.Sequential() #sequential which is basic nn
# #adding layers
# model.add(tf.keras.layers.Flatten(input_shape = (28,28))) #flatten layer
# model.add(tf.keras.layers.Dense(128, activation= 'relu')) #"RELU rectified  linear unit
# model.add(tf.keras.layers.Dense(128, activation= 'relu')) #"RELU rectified  linear unit
# model.add(tf.keras.layers.Dense(10, activation= 'softmax')) #"SOFTMAX IS MAKES ALL NEURONS ADD UPTO 1(HAVING SOME VALUE) SURE ALL output layer having 10  0-9 neurons
# #softmax gives the probability for digits to be correct
#
# #compiling the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#
# #fit the model train the model
# model.fit(x_train,y_train, epochs=3) #model is ready "epochs is how many iteration im gonna see
#
# model.save('handwritten.model')
#
# instead of runnig all this again and again i can comment it down and follwing the below way
#
# loading the model the i created because it is created one check the folder handwritten.model

#YAHAN COMMENT MAT KAR
model = tf.keras.models.load_model('handwritten.model')
#
# loss, accuracy = model.evaluate(x_test,y_test)

# print(loss)
# print(accuracy)

#comment out everything
#and read images from the folder

image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        #not interested in colors so getting last channel
        img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"this digit is probably a {np.argmax(prediction)}")
        #np.argmax(prediction) it gives the index of field that has highest number
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        image_number += 1







