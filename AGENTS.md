# AGENTS.md — SpeculativeAI Exp. #1, `master` branch

Context for coding agents (and humans). This branch is a different,
self-contained architecture from `SAI-Performance`/`fix-artnet`/
`without_replay` — don't assume they're interchangeable.

## What's on this branch

One large commit ("trained model for second version of Interspace",
2022-05-25) added everything at once: `main.py`, the whole `conversation/`
package, two trained models (`model_object_1.h5`, `model_object_2.h5`),
`archive/` (older experiments, including `model_object_2.h5` again and a
`src_lstm.py` — an unrelated LSTM experiment, see Exp. #2 below, don't
confuse with this feed-forward pipeline), `create_optimized_model.py`
(TensorRT export, historically for deploying to an NVIDIA Jetson TX2),
`trngsdata.txt` (308 training samples), and `teensyLEDController/` (a
separate Arduino/Teensy + FastLED/OctoWS2811/Artnet firmware project — not
verified during this session whether/how it relates to the LED controllers
actually in use; treat as a separate, uninvestigated subsystem).

Both `README.md` files in this repo (this branch and `SAI-Performance`) are
stale 2019/2020-era descriptions of the *other* (30-bin, Ableton-driven)
pipeline — they do not describe what's actually on this branch.

## The real signal chain here — no Ableton, no OSC from a DAW

```
Microphone (or a virtual audio cable, e.g. VB-Audio Virtual Cable, if you
want to feed it from Ableton/any DAW without a physical mic)
   -> conversation/fft.py (PyAudio, Hanning window, np.fft.rfft, 32
      non-linear bands starting at 60 Hz: endFreq = int(startFreq*1.12+12))
   -> conversation/neuralnet_audio.py (loads model_object_1.h5 or _2.h5,
      32 inputs -> 13824 outputs)
   -> conversation/interspace_statemachine.py
   -> conversation/interspace_artnet.py  (sends ArtNet *directly* over UDP
      to the LED controllers -- no separate Java visualizer/ortlicht here)
```

`main.py` on this branch is just `InterspaceStateMachine()` run in a loop —
it does not listen on any OSC port itself; the audio capture in `fft.py` is
the input.

## The two models

`model_object_1.h5` and `model_object_2.h5`: both `Dense(512, sigmoid)` ->
`Dense(13824, sigmoid)`, 32 inputs, ~85M params, saved via `model.save()`
(full model incl. optimizer state, hence ~57MB each despite the modest
architecture). Per the artist (2026-09-24): these correspond to light
**object 1** and **object 2** respectively, used alongside the separate
Klangobjekt/Dodekaeder work in `SpeculativeAI_Dodeca`. They are the
TX2-optimized descendants of a 20-bin -> 30-bin -> 32-bin model lineage
(see `SAI-Performance`'s `AGENTS.md` for the full chronology) — do **not**
pair either of these with the Ableton-OSC/30-value chain from
`SAI-Performance`; they were trained for `fft.py`'s 32-band, 60Hz-start
feature scale, not Ableton's, and Ableton's real FFT features were never
validated against them.

**Loading either model on a modern stack needs `tf_keras`, not bare
`keras`.** Same issue as on `SAI-Performance`: `conversation/neuralnet_audio.py`
does `import keras` (line 7); on an environment where `keras` resolves to
standalone Keras 3.x (very likely, it's a common transitive dependency of
modern `tensorflow`), loading these HDF5 files and then calling the model
directly raises a shape-inference error (`Expected shape (None, None, 32)`).
Fix: use `tf_keras` instead, and load with `compile=False` (the saved
optimizer config uses the old `lr=` kwarg, rejected by modern optimizers).

`trngsdata.txt` (308 rows, 32 input columns + 13824 output columns) is
almost certainly the training data for one of the two models above — not
independently verified which one, or whether both were trained from it.

## Do not confuse with

- **`SpeculativeAI_Dodeca`** (separate GitHub repo): the sound/Klangobjekt
  side. Its `dictionary_model_*.h5` files (128 inputs) are unrelated to
  Interspace's LED prediction, despite similarly-named/-shaped files
  turning up in old local backups (`D:\modelle`, `plane_and_sphere`-named
  experiments, also 128 inputs -> 13824 outputs) — those are a third,
  separate lineage (likely an early geometric/non-audio bootstrap phase),
  not usable with either the Ableton or the microphone pipeline.
- **Exp. #2** (LSTM/RNN, `archive/src_lstm.py`, and `model_RNN.h5` /
  `model_sw_sound_01.h5` found in old backups): a different, sequence-based
  experiment. Needs a `(batch, timesteps, features)` input and stateful
  prediction across frames — not a drop-in replacement for the feed-forward
  pipelines described above.
