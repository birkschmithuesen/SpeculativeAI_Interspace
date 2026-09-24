# AGENTS.md — SpeculativeAI Exp. #1, `SAI-Performance` branch

Context for coding agents (and humans) working on this branch. See
`README.md` for the original 2019/2020 project description; this file
documents what a 2026-09 debugging/retraining session (Tracking-Laptop)
found, so the next person doesn't have to re-derive it.

## What this branch actually contains

Only three tracked files: `README.md`, `main.py`, `requirements.txt`. No
model file, no `conversation/` package, nothing else — a working live
performance needs a compatible `model.h5` and a modern Python environment
sourced/built separately.

## The real signal chain (as run 2026-09, with Ableton)

Ableton Live → Max for Live device chain (`fftBinsToOSC.amxd` → internal
`fftBinsToList` sub-patch, `pfft~` FFT, non-linear magnitude bins →
`OSC-Sender` device builds `/fft` with N float values) → OSC to Python
`main.py`, port 8000 → `model.predict()` → UDP to `ortlicht` (separate
Java/Processing repo), port 10005, 10 packets of 1402 bytes, header
`!IBB(frameCount, index, 0)`, byte-encoded with a `-127` offset → ArtNet →
LED controllers, plus an on-screen 3D preview in ortlicht (toggle with the
`d` key).

**The Max device's own tooltip is stale.** It says "20 first bins of a 1024
fft size", but the actually-embedded, live-edited device (found by
comparing the on-disk `Max7/` reference copies against the Ableton
project's own "Imported" device cache, both dated 2018-08-13) uses `uzi
30` — sends 30 real, independent values, not 20 zero-padded ones. The FFT
overlap factor was also bumped from `1024 2` to `1024 4` at some point.
Don't trust the tooltip; if in doubt, open the live device and check the
`uzi`/`pack` objects directly.

## Model history (confirmed with the artist, 2026-09-24)

1. **20-bin** (`model_PERCEPTRON.h5`/`.json`, Jan/Feb 2018): `Dense(6144,
   relu)` → `Dropout` → `Dense(13824, sigmoid)`. Referenced (commented out)
   in this branch's `main.py`, superseded before this branch's active code
   path was written.
2. **30-bin** (this branch): `Dense(6144, sigmoid)` → `Dense(13824,
   sigmoid)`, matches `archive/createModelSaveToDisk.py`. This is what
   `main.py`'s hardcoded `fft_handler`/prediction loop actually expects.
3. **32-bin, optimized for an NVIDIA Jetson TX2 deployment** (branches
   `fix-artnet`, `without_replay`, `master`): `Dense(512, sigmoid)` →
   `Dense(13824, sigmoid)`, 12x smaller hidden layer. Paired with a
   completely different, self-contained pipeline (`conversation/fft.py`
   reading a microphone directly via PyAudio, no Ableton/OSC at all — see
   `master`'s `AGENTS.md`). **Do not pair a 32-input model with this
   branch's Ableton chain** — wrong bin count, wrong frequency scale, and
   Ableton's real FFT features were never validated against it.

**No trained 30-bin model.h5 survives** in this repo, in its GitHub
Releases/tags, or in the Nextcloud project backup checked during this
session (`260921_SAI/SAI_LivePerformance_01 Project/Modelle`, which only
had 20-bin and a family of unrelated 128-bin models — see below). The
original download link in `README.md`
(`www.birkschmithuesen.com/SAI/model.h5`) is dead.

A **new 30-bin model was trained 2026-09-24** using this branch's own
architecture and real (if small) training data found on an external backup
drive: `D:\modelle\OLD_models\traingsdata_standart.txt` (1728 samples, not
in this repo). 250 epochs, batch_size=128, `SGD(learning_rate=0.06,
momentum=0.9, nesterov=True)`, `binary_crossentropy`, final loss 0.0742 (at
zero input, mean brightness dropped from ~29% for the era's only other
30-input checkpoint we could find to ~5% for this one — the old checkpoint
looks undertrained: only 308 training samples, epochs applied live/
interactively via OSC `/train` calls). Too large for a normal commit (681MB,
this repo has no Git LFS) — published as a GitHub Release instead, see
`model-30bin-retrained-2026-09-24` in this repo's Releases tab.

Unrelated 128-input models also turned up in the same local backup
(`D:\modelle`, `LAST_model.h5`, `model_01_..._plane_and_sphere_*.h5`) — do
not confuse with the above; different input size, likely an early
geometric/non-audio bootstrap phase, output size (13824) matches by
coincidence.

## Running `main.py` on a modern Python/TensorFlow stack

The original code needs four fixes to run at all, found empirically (not
committed back into `main.py` as written here — reapply if you need to run
it):

1. **`import keras` crashes loading this model.** Modern environments
   commonly have both `keras` (standalone Keras 3.x) and `tf_keras` (the
   TensorFlow-bundled Keras-2-compatible package) installed. Keras 3's
   legacy-HDF5 loader mis-infers this `Sequential` model's input rank and
   raises `Expected shape (None, None, 30), but input has incompatible
   shape (1, 30)` on the very first prediction. Fix: `import tf_keras as
   keras` (and the equivalent submodule imports) instead of bare `keras`.
2. **`keras.models.load_model("model.h5")` needs `compile=False`.** The
   saved file embeds the original SGD optimizer config with the old `lr=`
   kwarg; modern Keras's optimizer deserialization rejects it
   (`ValueError: Argument(s) not recognized: {'lr': ...}`) unless you skip
   recompiling.
3. **`model.predict()` is far too slow for real time** — roughly 28ms/frame
   for this architecture, dominated by real compute rather than the
   ~56ms/frame Dataset/callback overhead you'd see on a smaller model (that
   overhead is what `model.predict_on_batch()` avoids; a direct
   `model(tensor, training=False)` / `tf.function` call is *not* faster for
   this model size, and can hit the same Keras-3-only shape bug as #1 —
   use `predict_on_batch()`, not a raw call, for this size of model).
4. **No backlog handling.** `fft = []` with `fft.pop(0)` is O(n) per pop,
   and there's no mechanism to drop stale data if Ableton sends faster than
   the model can keep up (it does: ~60 msg/s measured vs. ~35 fps max
   throughput at this model size). Without a fix the input queue grows
   without bound and the light show ends up seconds behind the real audio.
   Fix: `collections.deque()` + `popleft()`, plus trimming the queue down
   to a small buffer (e.g. the newest ~300 values / 10 frames) every time a
   new OSC packet arrives.

## `ortlicht` companion repo (separate GitHub repo `ortlicht`)

**Use branch `SAI-Performance`** (created 2026-09-24 from `master`, same
commit `40fc8ac`) for this performance, not `master` — it has the fixes
below already applied and built (`dist/ortlicht-neu.jar` is a build output,
not committed; run `build_ortlicht.bat` after checkout). `master` stays the
general/shared line.

Three things fixed there during this session, relevant if the light object
still looks wrong even with a correct model:

- **LED calibration data was stale.** A working checkout's
  `ortlicht/data/ledPositions.txt` and `regressionNormals.txt` did not
  match the `object1` calibration in the `ortlicht` repo's `SAI_training`
  branch (`data/object1/ledPositions.txt` etc. — identical to `ortlicht`'s
  `master`-branch top-level copy, and to `SAI-Performance`'s). If the
  object in use is object1 (confirmed with the artist for this session),
  make sure a working checkout's copy matches those, they're easy to
  accidentally overwrite with a stale local calibration.
- **The on-screen 3D preview was disabled.** `Ortlicht.java draw()` had the
  `drawScreen()` call commented out. This does *not* affect the real
  ArtNet output (`artNetSender.sendToLeds()` is called before the preview
  draw either way) — only the on-screen visualization was dark. Re-enabled,
  with a `d` key toggle and the previously-unthrottled `"get NN"`
  per-frame heartbeat log throttled to once/second.
- **`/NN/play` used to take an exclusive shortcut around the mixer.**
  `draw()` called `nnListener.getFrame()` directly and skipped
  `mixer.mix()` entirely whenever NN data was flowing, even though `NNefx`
  already exists as a proper mixer effect reading the same
  `nnListener.getFrame()` data — so the mixer's other effects
  (`MovingWallEffect`, `ManualSphere`, `VideoPlayer`, ...) froze/
  disappeared as soon as SAI Python started sending. `draw()` now always
  calls `mixer.mix()`; the NN layer blends in via `NNefx`'s own
  `/mixer/opacity/neuralNetwork` and `/colors/NN/` parameters (set by
  `nnplay.py`) instead of an exclusive on/off switch. Note: `/NN/play`
  itself no longer gates visibility — only the throttled `"get NN"` log
  line still reads it, purely as a diagnostic.
