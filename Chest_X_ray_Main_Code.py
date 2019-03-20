### Chest X-ray (Pneumonia) Classification Problem
### Download data from: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
### Unzip the data and place this code in the same directory as the directory with the data (complete folder)

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import datetime, time, random, joblib
import os, math, sys, glob
import operator as oper
import itertools as it
import functools as fnc
import seaborn as sns
from matplotlib import pyplot as plt
import tensorflow as tf
import keras
from keras.models import Sequential, Input, Model, load_model 
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, SeparableConv2D, GlobalMaxPool2D
from keras.layers.normalization import BatchNormalization as BatchNorm
from keras.preprocessing import image
from keras import backend as K
from keras.optimizers import SGD, Adam, RMSprop
from keras.preprocessing.image import img_to_array, ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import keras.metrics as metrics
from keras.callbacks import TensorBoard
import sklearn.utils as util

import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder as lab_enc
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import pathlib, cv2

random_seed = 2019
tf.set_random_seed(random_seed)

from tensorflow.python.client import device_lib
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type=='GPU']

print(get_available_gpus())

fileloc = os.path.realpath(__file__)
maindir = os.path.dirname(fileloc)

savedir = os.path.join(maindir, "Saved_Data_Joblib")
if not os.path.exists(savedir):
    os.mkdir(savedir)
    
modeldir = os.path.join(maindir, "Saved_Models")
if not os.path.exists(modeldir):
    os.mkdir(modeldir)

logdir_tb = os.path.join(maindir, "TensorBoard_Log_Dir")
if not os.path.exists(logdir_tb):
    os.mkdir(logdir_tb)
    
### find the directory with the images
### the main data folder should contain the following three folders
data_folder = ["train", "test", "val"] 
folders = os.listdir(maindir)
for f in folders:
    temp = os.path.join(maindir, f)
    if os.path.isdir(temp) and os.listdir(temp)!=[] and \
       all(i in os.listdir(temp) for i in data_folder):
        dataloc = temp
del temp, data_folder, folders, f

train_folder = os.path.join(dataloc, "train")
test_folder = os.path.join(dataloc, "test")
val_folder = os.path.join(dataloc, "val")

#### create a data frame of filepaths and targets. we will use flow_from_dataframe image generator in keras
def create_df(datadir, dataname):
    joblib_data = os.path.join(savedir, "Chest_Xray_" + dataname + "_images.joblib")
    if glob.glob(joblib_data) == []:
        df = pd.DataFrame(data=None, columns=["Filename", "Filepath", "Target_1", "Target_2"])
        colnames = df.columns
        foldername = [i for i in os.listdir(datadir) if i != ".DS_Store"]
        for folder in foldername:
            df_temp = pd.DataFrame(data=None, columns=["Filename", "Filepath", "Target_1", "Target_2"])
            print("Dealing with " + dataname + " folder: " + folder)
            filename = [i for i in os.listdir(os.path.join(datadir,folder)) if i.endswith(".jpeg")]
            for i in range(len(filename)):
                df_temp.loc[i, colnames[0]] = filename[i]
                df_temp.loc[i, colnames[1]] = os.path.join(os.path.join(datadir,folder), filename[i])
                df_temp.loc[i, colnames[2]] = folder
                if folder == "NORMAL":
                    df_temp.loc[i, colnames[3]] = folder.lower()
                else:
                    f = {"bacteria", "virus"} & set(filename[i].split("_"))
                    df_temp.loc[i, colnames[3]] = f.pop().lower()

            df = pd.concat([df, df_temp], join="outer")
            df.reset_index(drop=True, inplace=True)

        joblib.dump(df, joblib_data)
        return df
    else:
        df = joblib.load(joblib_data)
        return df

train_df = create_df(train_folder, "train")
test_df = create_df(test_folder, "test")
val_df = create_df(val_folder, "val")

### Choose either of the two target columns
target_name = "Target_2"  
num_class = len(train_df[target_name].unique())
INPUT_SHAPE = (224, 224, 1)

### to be used in model.fit function
sample_weights = util.compute_sample_weight("balanced", train_df[target_name])

model_name_prefix = "Chest_Xray_Classify_Model_"
#### Storing trained models...different names when training for different number of classes
if target_name == "Target_1":
    model_name = model_name_prefix + "normal_pneumonia.h5"
    model_path = os.path.join(modeldir, model_name)
elif target_name == "Target_2":
    model_name = model_name_prefix + "normal_bacteria_virus.h5"
    model_path = os.path.join(modeldir, model_name)


def model_build(image_shape=INPUT_SHAPE, classes=num_class):
    input_img = Input(shape=INPUT_SHAPE, name="Image_Input")
    x = Conv2D(64, (5,5), strides=(1,1),\
               padding="valid", name="Conv_1_1")(input_img)
    x = BatchNorm(name="BN_1_1")(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(3,3), padding="valid", name="Pool_1_1")(x)

    x = Conv2D(128, (3,3), strides=(1,1), padding="valid", name="Conv_2_1")(x)
    x = BatchNorm(name="BN_2_1")(x)
    x = Activation("relu")(x)
    x = Conv2D(128, (3,3), strides=(1,1), padding="valid", name="Conv_2_2")(x)
    x = BatchNorm(name="BN_2_2")(x)
    x = Activation("relu")(x)
    x = Conv2D(128, (3,3), strides=(1,1), padding="valid", name="Conv_2_3")(x)
    x = BatchNorm(name="BN_2_3")(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2,2), padding="valid", name="Pool_2_1")(x)

    x = Conv2D(256, (3,3), strides=(1,1), padding="valid", name="Conv_3_1")(x)
    x = BatchNorm(name="BN_3_1")(x)
    x = Activation("relu")(x)
    x = Conv2D(256, (3,3), strides=(1,1), padding="valid", name="Conv_3_2")(x)
    x = BatchNorm(name="BN_3_2")(x)
    x = Activation("relu")(x)
    x = Conv2D(256, (3,3), strides=(1,1), padding="valid", name="Conv_3_3")(x)
    x = BatchNorm(name="BN_3_3")(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2,2), name="Pool_3_1")(x)

    x = Conv2D(512, (3,3), strides=(1,1), padding="valid", name="Conv_4_1")(x)
    x = BatchNorm(name="BN_4_1")(x)
    x = Activation("relu")(x)
    x = Conv2D(512, (3,3), strides=(1,1), padding="valid", name="Conv_4_2")(x)
    x = BatchNorm(name="BN_4_2")(x)
    x = Activation("relu")(x)
    x = Conv2D(512, (3,3), strides=(1,1), padding="valid", name="Conv_4_3")(x)
    x = BatchNorm(name="BN_4_3")(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2,2), name="Pool_4_1")(x)

    x = Flatten(name="Flatten")(x)
    x = Dense(1024, activation="relu", name="FC_1")(x)
    x = Dense(256, activation="relu", name="FC_2")(x)
    x = Dense(classes, activation="softmax", name="FC_3")(x)

    model = Model(inputs=input_img, outputs=x)
    return model

#### train a model if training file not available
if not os.path.exists(model_path):
    print("Model file {} ABSENT, Training new model...".format(model_name))
    model = model_build()
    print(model.summary())
    
    epochs = 100
    batch_size = 12
    steps_epoch = len(train_df)//batch_size
           
##    rmsprop = RMSprop(lr=0.001, rho=0.90, epsilon=1e-08, decay=0.0)
    model.compile(optimizer="adam", loss = "categorical_crossentropy",metrics=["acc"])

    #### Callbacks for the model
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3,\
                                                verbose=1, factor=0.8, min_lr=1e-7)
    
    earlystop = EarlyStopping(monitor='val_acc', mode='max', verbose=0, patience=20)
    modelcheckpt = ModelCheckpoint(model_path, monitor='val_acc', mode='max',\
                                   verbose=1, save_best_only=True)
    tensorboard = TensorBoard(log_dir = logdir_tb, histogram_freq=0, write_graph=True, \
                              write_images=True, update_freq="epoch")
    tensorboard.set_model(model)

    #### defining a custom generator and finding sample weights for the training data
    train_img_datagen = ImageDataGenerator(fill_mode="constant", cval=0.0, horizontal_flip=True, \
                                           vertical_flip=False, height_shift_range=(-0.1,0.1), \
                                           brightness_range=(0.75, 1.25), rotation_range=10,\
                                           width_shift_range=(-0.1,0.1), shear_range=0.1, zoom_range=(0.75,1.25),\
                                           rescale=(1./255), dtype=np.float32)
    sample_weights = util.compute_sample_weight("balanced", train_df[target_name])
    
    train_datagen = train_img_datagen.flow_from_dataframe(train_df, directory=None, x_col="Filepath", \
                                                    y_col=target_name, target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]),\
                                                    color_mode="grayscale", class_mode="categorical",\
                                                    batch_size=batch_size, shuffle=True, \
                                                    seed=random_seed, interpolation="bicubic")

    test_img_datagen = ImageDataGenerator(rescale=(1./255), dtype=np.float32, fill_mode="constant", cval=0.0)
    test_datagen = val_img_datagen.flow_from_dataframe(test_df, directory=None, x_col="Filepath", \
                                                    y_col=target_name, target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]),\
                                                    color_mode="grayscale", class_mode="categorical",\
                                                    batch_size=batch_size, shuffle=True, \
                                                    seed=random_seed, interpolation="bicubic")

    history = model.fit_generator(train_datagen, steps_per_epoch=steps_epoch, epochs=epochs, verbose=2,\
                        class_weight=sample_weights, initial_epoch=0, shuffle=True,\
                        validation_data=test_datagen, validation_steps=steps_epoch, \
                        callbacks = [learning_rate_reduction, earlystop, modelcheckpt, tensorboard])
   
    
else:
    print("Model file {} PRESENT, Loading trained model...".format(model_name))
    model = keras.models.load_model(model_path, compile=True)







                
