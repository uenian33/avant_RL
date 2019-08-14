# Code to train T3D model
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam, SGD
import keras.backend as K
import traceback
import cv2
from matplotlib import pyplot as plt
import random
from glob import glob

from T3D_keras import densenet161_3D_DropOut, densenet121_3D_DropOut, xception_classifier, c3d_model, c3d_model_feature

import avant_para
# there is a minimum number of frames that the network must have, values below 10 gives -- ValueError: Negative dimension size caused by subtracting 3 from 2 for 'conv3d_7/convolution'
# paper uses 224x224, but in that case also the above error occurs
FRAMES_PER_VIDEO = 5
FRAME_HEIGHT = 112
FRAME_WIDTH = 112
FRAME_CHANNEL = 3
NUM_CLASSES = 50
BATCH_SIZE = 30
EPOCHS = 5
MODEL_FILE_NAME = 'T3D_saved_model.h5'

models_name = ['t3d', 'xception', 'c3d', 'c3d_feature']


def get_visual_states_2models(image_sequence):
     # Set classifier classes: bad, good
    nb_classes = 2

    pretrained_name = 'weights/visual_weights/C3D_feature_saved_model_weights.hdf5'
    save_name = 'weights/visual_weights/C3D_feature_saved_model.h5'
    sample_input = np.empty(
        [frames_sample, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL], dtype=np.uint8)
    model = c3d_model_feature(sample_input.shape, nb_classes)  # , feature=True)
    # compile model
    optim = Adam(lr=1e-4, decay=1e-6)
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
    # load pretrained weights
    if os.path.exists('./' + pretrained_name):
        print('Pre-existing model weights found, loading weights.......')
        model.load_weights('./' + pretrained_name)
        print(pretrained_name)
        print('Weights loaded')

    pretrained_name = 'weights/visual_weights/C3D_saved_model_weights.hdf5'
    save_name = 'weights/visual_weights/C3D_saved_model.h5'
    sample_input = np.empty(
        [frames_sample, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL], dtype=np.uint8)
    model2 = c3d_model(sample_input.shape, nb_classes)
    optim = Adam(lr=1e-4, decay=1e-6)
    model2.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
    # load pretrained weights
    if os.path.exists('./' + pretrained_name):
        print('Pre-existing model weights found, loading weights.......')
        model2.load_weights('./' + pretrained_name)
        print(pretrained_name)
        print('Weights loaded')

    #cap = cv2.VideoCapture(path)

    f, pred = model.predict(image_sequence)  # f is the feature array, pred is the prediction value
    pred2 = model2.predict(image_sequence)

    reward = 0
    if pred[0][1] < 0.5 and pred2[0][1] < 0.489:  # pred[0][0] > pred[0][1]:
        reward = -1
    else:
        reward = 1

    return f, reward


def get_visual_states_1model(image_sequence):
     # Set classifier classes: bad, good
    nb_classes = 2

    pretrained_name = 'weights/visual_weights/C3D_feature_saved_model_weights.hdf5'
    save_name = 'weights/visual_weights/C3D_feature_saved_model.h5'
    sample_input = np.empty(
        [frames_sample, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL], dtype=np.uint8)
    model = c3d_model_feature(sample_input.shape, nb_classes)  # , feature=True)
    # compile model
    optim = Adam(lr=1e-4, decay=1e-6)
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
    # load pretrained weights
    if os.path.exists('./' + pretrained_name):
        print('Pre-existing model weights found, loading weights.......')
        model.load_weights('./' + pretrained_name)
        print(pretrained_name)
        print('Weights loaded')

    #cap = cv2.VideoCapture(path)

    f, pred = model.predict(image_sequence)  # f is the feature array, pred is the prediction value

    reward = 0
    if pred[0][1] < 0.5:
        reward = -1
    else:
        reward = 1

    return f, reward


def cal_states_rewards(sensor_data, image_sequence, mode):
    """
        calculate states and rewards
    """
    s_data = np.asarray(sener_data)
    im_featre, reward = get_visual_states_1model(image_sequence)  # or get_visual_states_2models(image_sequence)
    if obs_mode == avant_para.state_modes[0]:
        # 8 vis + 4 senor states: Angle boom, Angle bucket, Length telescope, TransmissionPressureSensor16 -TransmissionPressureSensor13
        s = np.hstack((im_featre, s_data))
    elif obs_mode == avant_para.state_modes[1]:
        # 4 senor states: Angle boom, Angle bucket, Length telescope, TransmissionPressureSensor16 -TransmissionPressureSensor13
        s = np.hstack((s_data))
    elif obs_mode == avant_para.state_modes[0]:
        s = np.hstack((im_featre))  # 8 vis
    else:
        s = np.hstack((im_featre, s_data))
    return s, reward


def reset_avant():
    """
        send hard code command to avant,
        make it back to the start point,
        get sensor data, and frames
    """
    s_data, image_sequence = reset_action()
    s_data = np.asarray(sener_data)
    im_featre, reward = get_visual_states_1model(image_sequence)  # or get_visual_states_2models(image_sequence)
    s = np.hstack((im_featre, s_data))
    return s, reward


def command_to_avant(actions, mode):
    """
        send commands to avant 
        after sending then get the data
    """
    s_data, image_sequence = update_action()
    return s_data, image_sequence


def get_step_states(actions):
    s_data, image_sequence = command_to_avant(actions)
    return cal_states_rewards(s_data, image_sequence, mode)
