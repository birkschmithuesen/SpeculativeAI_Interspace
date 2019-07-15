"""
The class contains a neural net for predicting 13824 led
brightness values from an 30 bin fft vector input.
"""
import os
import keras
from keras.models import Sequential
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot
from keras.layers import Dense, LSTM, Dropout, Activation
from keras.optimizers import SGD
from keras.models import model_from_json
from keras.models import load_model
from keras import backend as kerasBackend
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #force Tensorflow to use the computed
LOAD_MODEL = True
SAVE_MODEL = False

MODEL_FILE_PATH = './model.h5'
MODEL_TRAININGS_DATA_FILE_PATH = 'traingsdata.txt'
MODEL_SAVE_FILE_PATH = './model.h5'

INPUT_DIM = 128
BATCH_SIZE = 32
EPOCHS = 30
INITIAL_EPOCHS = 150

HIDDEN1_DIM = 512
HIDDEN2_DIM = 4096
OUTPUT_DIM = 13824

model = Sequential()

def load_model_from_file():
    """
    Load model from file
    """
    global model
    model = load_model(MODEL_FILE_PATH)
    model._make_predict_function()
    print('Loaded saved model from file')

def train_model():
    """
    loading the trainingsdata from a textfile. Convert the
    trainingpoints in the right dimension: One line in the text file is
    one trainingpoint with 30 FFT values and 13824 LED values, separated by tabulators
    """
    global model
    #import fft and led input data
    file_name = MODEL_TRAININGS_DATA_FILE_PATH
    file = open(file_name)
    print('Loading Trainingsdata from File:', file_name,'  ...')
    values = loadtxt(file_name, dtype='float32')
    print('Trainingsdata points: ', values.shape[0], "\n")

    #split into input and outputs
    training_input, training_output = values[:,:-OUTPUT_DIM], values[:,INPUT_DIM:]
    print('training_input shape: ', training_input.shape, 'training_output shape: ', training_output.shape)
    my_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
    model.add(Dense(HIDDEN1_DIM, activation='sigmoid', input_dim=INPUT_DIM, kernel_initializer=my_init, bias_initializer=my_init))
    model.add(Dense(HIDDEN2_DIM, activation='sigmoid', input_dim=HIDDEN1_DIM, kernel_initializer=my_init, bias_initializer=my_init))
    model.add(Dense(OUTPUT_DIM, activation='sigmoid',kernel_initializer=my_init, bias_initializer=my_init))
    sgd = SGD(lr=0.06, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(training_input, training_output, epochs=INITIAL_EPOCHS, batch_size=32, shuffle=True)
    model._make_predict_function()
    model.summary()
    print('Loaded new model')

def new_model_handler(unused_addr, args):
    """
    this function should reinitialize the model, to start the training from scratch again.
    ToDo: make it work. Probably the crash is caused because of the multi-threading?
    """
    global model
    kerasBackend.clear_session()
    my_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)

    model.add(Dense(6144, activation='sigmoid', input_dim=30, kernel_initializer=my_init,
                    bias_initializer=my_init))
    model.add(Dense(OUTPUT_DIM, activation='sigmoid',kernel_initializer=my_init,
                    bias_initializer=my_init))
    sgd = SGD(lr=0.06, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(training_input, training_output, epochs=1, batch_size=32, shuffle=True)
    model._make_predict_function()
    print('Loaded new model')

def train_handler(unused_addr, args):
    """
    neural network trainer
    """
    global model
    model.fit(training_input, training_output, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True)
    model._make_predict_function()
    print('training finished...')
    print('')

def frame_count_handler(unused_addr, args):
    """
    a function to synchronize for recording the output of the neural network
    """
    #print('received frameCoount: ', args)
    global frame_count
    frame_count = args

def run():
    """
    Executes model loading/training/saving
    """
    global model
    if LOAD_MODEL:
        load_model_from_file()
    else:
        train_model()
    if SAVE_MODEL:
        model.save(MODEL_SAVE_FILE_PATH)
        print('Saved new model to path: ', MODEL_SAVE_FILE_PATH)
        model.summary()

if __name__ == "__main__":
    run()
