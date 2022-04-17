# SpeculativeArtificialIntelligence / Exp. #1
# (audiovisual associations)
The program generates a model of feed-forward NN with 30 inputs and 13824 outputs.<br>
30 inputs: FFT analyses of sounds<br>
13824 outputs: sets the brightness for 13824 LEDs of the lightobject "Interspace #3"<br>

Find a documentation about this work on: https://vimeo.com/280350114

The training data can be downloaded here and opened with liveTraining.py:
www.birkschmithuesen.com/SAI/traingsdata.txt

A trained model is here and opened with loadModel.py:
www.birkschmithuesen.com/SAI/model.h5

The program predicts the output for the lightobject in real time from received FFT data.
The communication is done via OSC.<br>
WARNING: if you have no network card with the fixed IP 2.0.0.1 in your computer, the program will crash. <br>
Also see: line 153 and line 102

# communication diagram:
Ableton Live(sound program)/FFT analysis => 30 float values via OSC (NN input) => python/neural network => 13824 float values via OSC (NN output) => JAVA(visualizer on screen and light object)

## packages/files needed
* TensorFlow (pip tensorflow)
* Keras (pip keras)
* PythonOsc (pip python-osc)
www.birkschmithuesen.com/SAI/traingsdata.txt

# installation

  * brew install portaudio
  * pip3 install -r requirements.txt
  
# installation for training
  * python=3.6.8
  * create Conda Environment with libraries defined in "specs-conda.txt": conda create --name $ENV_name --file specs-conda.txt
  * update tensorflow with pip according to "cpecs-pip.txt": pip install tensorflow==1.14
