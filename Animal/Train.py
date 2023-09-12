
import cv2
from glob import glob
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten,Dense,Conv3D,MaxPool3D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

anim_path=pathlib.Path(r"C:\Users\user\Desktop\CNN_Project\Animal\Data")

zero_list=list(anim_path.glob("wolf/*.png"))          # *-shows all #(malignant/*)- shows all files inside the malignant folder
one_list=list(anim_path.glob("tiger/*.png"))
two_list=list(anim_path.glob("lion/*.png"))
three_list=list(anim_path.glob("hyena/*.png"))
four_list=list(anim_path.glob("fox/*.png"))
five_list=list(anim_path.glob("cheetah/*.png"))
print(zero_list,"kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk")

anim_dict={"wolf":zero_list,
              "tiger":one_list,
              "lion":two_list,
              "hyena":three_list,
              "fox":four_list,
              "cheetah":five_list
             }
anim_class={"wolf":0,
              "tiger":1,
               "lion":2,
               "hyena":3,
               "fox":4,
               "cheetah":5

              }

x=[]
y=[]

for i in anim_dict:
  anim_name=i
  anim_path_list=anim_dict[anim_name]
  for path in anim_path_list:
    img=cv2.imread(str(path))
    img=cv2.resize(img,(224,224))
    img=img/255
    x.append(img)
    cls=anim_class[anim_name]
    y.append(cls)

len(x)

x=np.array(x)
y=np.array(y)

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.75,random_state=1)

len(xtrain),len(ytrain),len(xtest),len(ytest)

xtrain.shape

xtrain.shape,xtest.shape

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

base_model = tf.keras.applications.MobileNetV2(input_shape=(32, 32, 3),include_top=False,weights='imagenet')

print("[INFO] summary for base model...")
print(base_model.summary())

from tensorflow.keras.layers import MaxPooling2D
from keras.layers.core import Dropout
from tensorflow.keras import datasets, layers, models
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(6,activation='softmax'))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(xtrain, ytrain, epochs=10,
                    validation_data=(xtest, ytest))

#model_hist=model.fit(xtrain,ytrain,epochs=5,validation_data=(xtest,ytest),batch_size=180)

model.save("weight.h5")

# from tensorflow.keras.preprocessing import image
# # testing the model
# def testing_image(image_directory):
#     test_image = image.load_img(image_directory, target_size = (224, 224))
#     test_image = image.img_to_array(test_image)
#     test_image = np.expand_dims(test_image, axis = 0)
#     test_image = test_image/255
#     result = model.predict(x= test_image)
#     print(result)
#     if np.argmax(result)  == 0:
#       prediction = 'wolf'
#     elif np.argmax(result)  == 1:
#       prediction = 'tiger'
#     elif np.argmax(result)  == 2:
#       prediction = 'lion'
#     elif np.argmax(result)  == 3:
#       prediction = 'hyena'
#     elif np.argmax(result)  == 4:
#       prediction = 'fox'

#     else:
#       prediction = 'cheetah'
#     return prediction

# print(testing_image('/content/drive/MyDrive/Python/Deep_Learning/CNN/Vegetables_classification/Dataset/data/Radish/0222.jpg'))

# from sklearn.metrics import  confusion_matrix

# Y_pred = model.predict_generator(xtest)

# y_pred = np.argmax(Y_pred, axis=1)

# print('Confusion Matrix')

# c=confusion_matrix(ytest, Y_pred)
# #cm = confusion_matrix(np.where(ytest), Y_pred)

