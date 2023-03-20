import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pickle

import numpy as np
import cv2
import matplotlib.pyplot as plt

data_dir = 'Resources'
categories = ['caesal','daisy','eupho','lantana','llily']


data = []

def make_data():
    for category in categories:
        path = os.path.join(data_dir, category)  #dir to data thats is iteration of f1,,,f8
        label = categories.index(category)

        for img_name in os.listdir(path):
            image_path = os.path.join(path,img_name)
            image = cv2.imread(image_path)

#working fine loading is perfect so commenting it out
            # cv2.imshow('image1', image)
        #
            # break
        #
        # break
#converting the color and resizing the image and making the dictionary of data
            try:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (224,224))
# image to array coversion
                image = np.array(image, dtype=np.float32)

                data.append([image,label])

            except Exception as e:
                pass

    print(len(data))
    pik =open('data.pickle', 'wb')
    pickle.dump(data,pik)
    pik.close()
    #

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

make_data()
#
# #loading pickle file
def load_data():
     pick = open('data.pickle','rb')
     data = pickle.load(pick)
     pick.close()
#
     np.random.shuffle(data)
#
     feature = []
     labels = []
#
     for img, label in data:
         feature.append(img)
         labels.append(label)
# #convert feature  to array
     feature = np.array(feature, dtype=np.float32)
     labels = np.array((labels))
#
     feature = feature/255.0 #scaling values 0 to 1 bfour feeding them to nn model
#
     return [feature, labels]

# def load_images_from_folder(testdata):
#     imags = []
#     for filename in os.listdir(testdata):
#         imgs = cv2.imread(os.path.join(testdata,))


#####DATA AUGMENTATION#########3
from keras.preprocessing.image import ImageDataGenerator
from skimage import io

# Construct an instance of the ImageDataGenerator class
# Pass the augmentation parameters through the constructor.

datagen = ImageDataGenerator(
rotation_range=45, #Random rotation between 0 and 45
width_shift_range=0.2, #% shift
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='reflect', cval=125)

#Manually read each image and create an array to be supplied to datagen via flow method
dataset = []


from skimage import io
import os
from PIL import Image

image_directory = 'Resources/penta/'
SIZE = 224
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
save_to_dir='Resources/penta',
save_prefix='aug',
save_format='jpg'):
i += 1
if i > 20:
break