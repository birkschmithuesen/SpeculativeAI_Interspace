from conversation import neuralnet_audio
import numpy as np
import random

neuralnet_audio.run()
print("RUUNING:")
prediction_input = np.asarray([[random.random() for i in range(neuralnet_audio.INPUT_DIM)]])
prediction_input.shape = (1, neuralnet_audio.INPUT_DIM)
prediction_output = neuralnet_audio.model.predict(prediction_input)
prediction_output = prediction_output.flatten()
print("PRDICTIONN:")
print(prediction_output)
