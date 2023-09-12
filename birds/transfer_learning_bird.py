# -*- coding: utf-8 -*-
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

bird_path=pathlib.Path(r"C:\Users\user\Desktop\cnn_dataset")

zero_list=list(bird_path.glob("Cape_Glossy_Starling/*.jpg"))          # *-shows all #(malignant/*)- shows all files inside the malignant folder
one_list=list(bird_path.glob("Cliff_Swallow/*.jpg"))
two_list=list(bird_path.glob("Common_Yellowthroat/*.jpg"))
three_list=list(bird_path.glob("Green_Jay/*.jpg"))
four_list=list(bird_path.glob("Horned_Puffin/*.jpg"))
five_list=list(bird_path.glob("Indigo_Bunting/*.jpg"))
six_list=list(bird_path.glob("Laysan_Albatross/*.jpg"))
seven_list=list(bird_path.glob("Red_legged_Kittiwake/*.jpg"))
eight_list=list(bird_path.glob("Scarlet_Tanager/*.jpg"))
nine_list=list(bird_path.glob("White_Pelican/*.jpg"))

len(zero_list),len(one_list),len(two_list),len(three_list),len(four_list),len(five_list),len(six_list),len(seven_list),len(eight_list),len(nine_list)

bird_dict={"Cape_Glossy_Starling":zero_list,
              "Cliff_Swallow":one_list,
              "Common_Yellowthroat":two_list,
              "Green_Jay":three_list,
              "Horned_Puffin":four_list,
              "Indigo_Bunting":five_list,
              "Laysan_Albatross":six_list,
              "Red_legged_Kittiwake":seven_list,
              "Scarlet_Tanager":eight_list,
              "White_Pelican":nine_list,
             }
bird_class={"Cape_Glossy_Starling":0,
              "Cliff_Swallow":1,
               "Common_Yellowthroat":2,
               "Green_Jay":3,
               "Horned_Puffin":4,
               "Indigo_Bunting":5,
               "Laysan_Albatross":6,
               "Red_legged_Kittiwake":7,
               "Scarlet_Tanager":8,
               "White_Pelican":9,

              }

x=[]
y=[]

for i in bird_dict:
  bird_name=i
  bird_path_list=bird_dict[bird_name]
  for path in bird_path_list:
    img=cv2.imread(str(path))
    img=cv2.resize(img,(224,224))
    img=img/255
    x.append(img)
    cls=bird_class[bird_name]
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

base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),include_top=False,weights='imagenet')

print("[INFO] summary for base model...")
print(base_model.summary())

from tensorflow.keras.layers import MaxPooling2D
from keras.layers.core import Dropout
from tensorflow.keras.models import Model
headModel= base_model.output
headModel= MaxPooling2D(pool_size=(2,2))(headModel)
headModel= Flatten(name="flatten")(headModel)
headModel= Dense(32, activation="relu")(headModel)
headModel= Dropout(0.2)(headModel)
headModel= Dense(10, activation="softmax")(headModel)
model= Model(inputs=base_model.input, outputs=headModel)
for layer in base_model.layers:
  layer.trainable= False

from tensorflow.keras.optimizers import Adam
print("[INFO] compiling model....")
opt= Adam(lr=1e-4)
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
print("[INFO] training head....")
model_hist=model.fit(xtrain,ytrain,epochs=10,validation_data=(xtest,ytest),batch_size=180)

model.save("weight.h5")


# from sklearn.metrics import  confusion_matrix

# Y_pred = model.predict_generator(xtest)

# y_pred = np.argmax(Y_pred, axis=1)

