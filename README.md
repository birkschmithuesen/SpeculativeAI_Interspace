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
TensorFlow (pip tensorflow)
Keras (pip keras)
PythonOsc (pip python-osc)
www.birkschmithuesen.com/SAI/traingsdata.txt

# installation

  brew install portaudio
  pip3 install -r requirements.txt

---

## 2026-09-24 update

A debugging/retraining session on the Tracking-Laptop found that the
original download links above are dead and no 30-input trained model
survives anywhere (this repo, its Releases, or checked backups). A new one
was trained from real training data found on a backup drive and published
as a GitHub Release (`model-30bin-retrained-2026-09-24`). See `AGENTS.md`
for that, the real (and outdated-tooltip) Ableton signal chain, four fixes
needed to run `main.py` on a modern Python/TensorFlow stack, and two
`ortlicht` bugs found along the way.
