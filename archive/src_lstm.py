import keras

# Hi Birk and Marcus,
# execute this code in order to see the architecture of the network

model = keras.models.Sequential()

INPUT_DIM = 30
NUM_SOUNDS = 60  # 2 seconds and 30 sounds per second?
LSTM_OUT = 512

HIDDEN1_DIM = 1024
OUTPUT_DIM = 13000

# input shape is (?, NUM_SOUNDS, INPUT_DIM): ? is the batch size, NUM_SOUNDS is the number of sounds in the
#                                            sequence and INPUT_DIM is the vector size of each sound
# output_shape is (?, LSTM_OUT): ? is the batch size and LSTM_OUT is the number of units in the output
# 'return_sequences' is False because you want only to codify NUM_SOUNDS sounds in one LSTM_OUT-dimensional vector
model.add(keras.layers.LSTM(units=LSTM_OUT, input_shape=(NUM_SOUNDS, INPUT_DIM),
                            return_sequences=False, name='lstm_layer'))

# This is a hidden layer. You can use it or not.
# In this case the activation can be ReLU. I write down 2048 output units but you can try other quantities
model.add(keras.layers.Dense(units=HIDDEN1_DIM, activation='relu', name='hidden1_layer'))

# the output layer must have 13000 units (one per led) and the activation has to be sigmoid
model.add(keras.layers.Dense(units=OUTPUT_DIM, activation='sigmoid', name='output_layer'))

# define the optimizer. You can use the optimizer that you want
adam = keras.optimizers.Adam(lr=0.0001)

# and finally use binary_crossentropy as loos function
model.compile(loss='binary_crossentropy', optimizer=adam)

model.summary()