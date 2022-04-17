"""
The class contains a neural net for predicting 13824 led
brightness values from an 30 bin fft vector input.

If there is a model.h5 file in the root, the model will be used.
If there is no pre trained model available, the system tries to load the trainingdata (trngsdata.txt) to create a new model
"""
import os
from tensorflow import Session, ConfigProto
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

# this seems to help to get the CUDNN with RTX cards to work
import keras.backend as K
cfg = K.tf.ConfigProto(gpu_options={'allow_growth': True})
K.set_session(K.tf.Session(config=cfg))

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #force Tensorflow to use the computed

MODEL_LOAD_FILE_PATH = '..\\model.h5'               #Linux / OSX: ./model.h5
MODEL_SAVE_FILE_PATH = '..\\model_new.h5'           #Linux / OSX: ./model_new.h5
MODEL_TRAININGS_DATA_FILE_PATH = '..\\trngsdata.txt'#Linux / OSX: ./trngsdata.txt

LOAD_MODEL = os.path.isfile(MODEL_LOAD_FILE_PATH)
SAVE_MODEL = not LOAD_MODEL

CONTINUE_TRAINING = False

INPUT_DIM = 32
BATCH_SIZE = 32
EPOCHS = 250
INITIAL_EPOCHS = 50
HIDDEN1_DIM = 512
#HIDDEN2_DIM = 4096
OUTPUT_DIM = 13824
LEARNING_RATE = 3.2

config = ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = Session(config=config)
model = Sequential()
training_input = 0
training_output = 0

def load_model_from_file():
    """
    Load model from file
    """
    global model
    model = load_model(MODEL_LOAD_FILE_PATH)
    print('Loaded saved model from file')

def build_model():
    """
    Initialize a new network
    """
    my_init=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
    model.add(Dense(HIDDEN1_DIM, activation='sigmoid', input_dim=INPUT_DIM, kernel_initializer=my_init, bias_initializer=my_init))
    #model.add(Dense(HIDDEN2_DIM, activation='sigmoid', input_dim=HIDDEN1_DIM, kernel_initializer=my_init, bias_initializer=my_init))
    model.add(Dense(OUTPUT_DIM, activation='sigmoid',kernel_initializer=my_init, bias_initializer=my_init))
    sgd = SGD(lr=LEARNING_RATE, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

def make_prediction_function():
    """
    Make the prediction function
    """
    global model
    model._make_predict_function()

def load_trainingsdata():
    """
    loading the trainingsdata from a textfile. Convert the
    trainingpoints in the right dimension: One line in the text file is
    one trainingpoint with 30 FFT values and 13824 LED values, separated by tabulators
    """
    global training_input, training_output
    #import fft and led input data
    file_name = MODEL_TRAININGS_DATA_FILE_PATH
    file = open(file_name)
    print('Loading Trainingsdata from File:', file_name,'  ...')
    values = np.loadtxt(file_name, dtype='float32')
    print('Trainingsdata points: ', values.shape[0], "\n")
    #split into input and outputs
    training_input, training_output = values[:,:-OUTPUT_DIM], values[:,INPUT_DIM:]
    print('training_input shape: ', training_input.shape, 'training_output shape: ', training_output.shape)


def train_model():
    """
    trains the model with INITIAL_EPOCHS
    """
    global model, training_input, training_output
    model.fit(training_input, training_output, epochs=INITIAL_EPOCHS, batch_size=BATCH_SIZE, shuffle=True)
    model._make_predict_function()
    model.summary()
    print('Initial training finished...')


def continue_training():
    """
    trains the model with EPOCHS
    """
    global model, training_input, training_output
    model.fit(training_input, training_output, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True)
    model._make_predict_function()
    print('training finished...')
    print('')

def save_model():
    """
    saves the model to disk
    """
    model.save(MODEL_SAVE_FILE_PATH)
    print('Saved new model to path: ', MODEL_SAVE_FILE_PATH)
    model.summary()

def run():
    """
    Executes model loading/training/saving
    """
    global model
    if CONTINUE_TRAINING:
        load_trainingsdata()
        load_model_from_file()
        continue_training()
        save_model()
    else:
        if LOAD_MODEL:
            load_model_from_file()
            make_prediction_function()
        else:
            build_model()
            load_trainingsdata()
            train_model()
    if SAVE_MODEL:
        save_model()

if __name__ == "__main__":
    run()
