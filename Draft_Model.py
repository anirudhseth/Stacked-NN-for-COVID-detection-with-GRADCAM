

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from data_module import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D,Conv2D,MaxPooling2D
from tensorflow.keras.layers import Dropout,Lambda
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
import tensorflow as tf


Traingenerator = BalanceCovidDataset(data_dir="/home/anirudh_seth/Dataset",
                                csv_file="/home/anirudh_seth/Dataset/train_split_v5.txt",
                                covid_percent=0.3,
                                batch_size=100,
                                is_training=True,
                                augmentation=False,
                                class_weights=[1., 1., 12]) #weights for normal, pneumonia, covid19



Valgenerator = BalanceCovidDataset(data_dir="/home/anirudh_seth/Dataset",
                                csv_file="/home/anirudh_seth/Dataset/val_split_v5.txt",
                                covid_percent=0.3,
                                batch_size=100,
                                is_training=False,
                                isValidation=True,
                                augmentation=False,
                                class_weights=[1., 1., 12]) #weights for normal, pneumonia, covid19


def checkDistribution(gen):
    for i in range(len(gen)):
        distribution=next(Traingenerator)[1]
        print('Batch ',i,' : ',np.sum(distribution,axis=0))



tensorboard_callback =tf.keras.callbacks.TensorBoard(
    log_dir='logs_exp3', histogram_freq=0, write_graph=True, write_images=False,
    update_freq='batch', profile_batch=2, embeddings_freq=0
)

checkpoint_filepath = 'checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_acc',
    mode='max',
    save_freq='epoch',
    save_best_only=False)




baseModel = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

headModel = baseModel.output
# headModel = AveragePooling2D(pool_size=(2, 2))(headModel)
headModel = Conv2D(512, (3, 3), activation=tf.nn.relu,name='block6_conv1')(headModel)
headModel = Conv2D(512, (3, 3), activation=tf.nn.relu,name='block6_conv2')(headModel)
headModel = MaxPooling2D(pool_size=(2, 2))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(3, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
	layer.trainable = False



opt = Adam(lr=0.001, decay=0.001 / 10)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
# train the head of the network
print("[INFO] training head...")
H = model.fit_generator(Traingenerator,validation_data=Valgenerator,verbose=1,epochs=50,callbacks=[tensorboard_callback,model_checkpoint_callback])


model.save('model_exp3_complete')

