#import os

import cv2
import numpy as np

#load YOLO

net = cv2.dnn.readNet("Resources/yolov3.weights", "Resources/yolov3.cfg")
classes = [] #we need them in array the names of things
with open("Resources/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()] #strip() rmv whitspaces

#print(classes)
#getting layers and output layers then we can get the detection of the object
layer_names = net.getLayerNames()
outputLayers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0,255,size=(len(classes), 3))
#loading image
img = cv2.imread("Resources/flower.jpg")
img = cv2.resize(img, None, fx=0.30, fy=0.90)
#giving track of original
height,width,channels = img.shape

#DETECTING OBJECTS
# # #to rgb blob is the way to extract the features from the image
blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416),(0,0,0), swapRB=True, crop=False)



# print(blob.shape)
# for b in blob:
#     for n, img_blob in enumerate(b): #ENUMRATE --> DIIFERENT NAME TO EACH WINDOW
#         cv2.imshow(str(n),img_blob) #TOSHOW (there will be 3 images r g b)



#passing blob image to algorithm (into the network)

net.setInput(blob)
outs = net.forward(outputLayers)

#showing info on the screen
class_ids = [] #putting rectangle values in array
confidences = []
boxes = []
for out in outs:
    for detection in out: #we need to detect confidence in 3 steps about surety
        scores = detection[5:] #index
        class_id = np.argmax(scores) #no associated with classes
        confidence = scores[class_id]
        if confidence > 0.5:
            #object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
        #   cv2.circle(img, (center_x, center_y), 10, (0, 255, 0), 2)
    # GREEN CIRCLE ON THE OBJTCS    0,255,0 green hain        2 thickness hain



            #RECTANGLE COORDINATES
            x = int(center_x - w / 2)  #topleft x
            y = int(center_y - h / 2)   #topleft y
            boxes.append([x,y,w,h])
            confidences.append(float(confidence)) #how confident
            class_ids.append(class_id)

#NMS non-max suppresion to have higher value objs
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
print(indexes)
#number_objects_detected = len(boxes)
font = cv2.FONT_ITALIC
for i in range(len(boxes)): #har object ko outp krna
    if i in indexes: #plays role to reduces boxes if various
         x, y, w, h = boxes[i]
         label = str(classes[class_ids[i]])
         color = colors[i]
    # print(label)
         cv2.rectangle(img, (x, y), (x + w, x + h), color, 2)
         cv2.putText(img, label,(x,y + 30), font, 3,color, 3)


cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

