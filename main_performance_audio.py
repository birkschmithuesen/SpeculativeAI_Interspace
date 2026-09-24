"""
SpeculativeAI - Performance-Kette mit direktem Audio-Zugriff statt OSC von Ableton.

Unterschied zu main_performance.py:
  * Kein OSC-Server auf Port 8000 mehr. Stattdessen wird der Sound direkt von
    einem Windows-Aufnahmegeraet abgegriffen, dessen Name "CABLE Output"
    enthaelt (VB-Audio Virtual Cable). Ableton muss seinen Ausgang manuell auf
    "CABLE Input" routen -- das ist bereits eingerichtet, nicht Teil dieses
    Skripts.
  * Die FFT-Berechnung (Hanning-Fenster, np.fft.rfft, 32 nicht-lineare Baender
    ab 60 Hz mit endFreq = int(startFreq*1.12+12)) ist 1:1 aus
    conversation/fft.py (Repo SpeculativeAI_Interspace) uebernommen, damit die
    32 Werte exakt zu model_object_1.h5 (32 Eingaenge) passen.

Unveraendert gegenueber main_performance.py (nicht anfassen):
  * Modell laden mit tf_keras, INPUT_DIM aus model.input_shape[1]
  * predict_fn per tf.function-Direktaufruf statt model.predict() (siehe dort:
    model.predict() kostet ~56 ms Verwaltung pro Frame, das ist fuer Echtzeit
    zu langsam)
  * led_output_loop: UDP-Ausgabe an ortlicht, Port 10005, 10 Pakete a 1402
    Byte, Header !IBB(frameCount, paketIndex, 0), Werte mit -127 Offset und
    Zweierkomplement-Maskierung (bereits gefixter Overflow-Bug)

Aufruf:
    venv\\Scripts\\python.exe main_performance_audio.py
"""

import argparse
import os
import socket
import struct
import sys
import threading
import time

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pyaudio

# ---------------------------------------------------------------- Konfiguration

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_object_1.h5")

UDP_IP = "127.0.0.1"            # ortlicht laeuft auf demselben Rechner
UDP_PORT = 10005
NUM_LEDS = 13824
PACKETS_PER_FRAME = 10
LEDS_PER_PACKET = 1402          # 10 * 1402 = 14020 >= 13824 (Original-Layout)

PRINT_EVERY = 30                # Statuszeile nur jede N-te Frame

# --- Audio / FFT, 1:1 aus conversation/fft.py (SpeculativeAI_Interspace) ---

FORMAT = pyaudio.paFloat32
RATE = 44100
FPS = 35
CHUNK = int(RATE / FPS)
WINDOW = np.hanning(CHUNK)
SPEC_X = np.fft.rfftfreq(CHUNK, d=1.0 / RATE).tolist()   # Frequenzachse, konstant bei festem CHUNK


def create_bins():
    """32 nicht-lineare Frequenzbaender, exakt wie createBins() in fft.py."""
    bins = []
    start_freq = 60
    for _ in range(32):
        end_freq = int(start_freq * 1.12 + 12)
        bins.append((start_freq, end_freq))
        start_freq = end_freq
    return bins


BINS = create_bins()


def closest_value_index(val, lst):
    """Index des ersten Listenwerts, der groesser als val ist (wie in fft.py)."""
    index = 0
    for item in lst:
        if item > val:
            return index
        index += 1
    return index - 1


def bin_spec_y(spec_y, start, end):
    """Mittelwert der FFT-Magnitude im Bereich [start, end] Hz (wie fft.py: bin_spec_y)."""
    start_idx = closest_value_index(start, SPEC_X)
    i = 0
    bin_sum = 0.0
    n = len(SPEC_X)
    while start_idx + i < n and SPEC_X[start_idx + i] <= end:
        bin_sum += spec_y[start_idx + i]
        i += 1
    return bin_sum / (i + 1)


def compute_fft_bins(samples):
    """Ein Audio-Chunk (float32, CHUNK Samples) -> 32 Frequenzbaender."""
    windowed = samples * WINDOW
    spec_y_raw = np.fft.rfft(windowed)
    spec_y = np.sqrt(spec_y_raw.real ** 2 + spec_y_raw.imag ** 2)
    return [bin_spec_y(spec_y, lo, hi) for lo, hi in BINS]


# ------------------------------------------------------------------- Zustand

fft_queue = []
fft_lock = threading.Lock()
frame_available = threading.Event()
output_queue = []
output_ready = threading.Event()

frame_count = 0
stats = {"received": 0, "predicted": 0, "sent": 0}
running = True

INPUT_DIM = None
model = None
predict_fn = None          # schneller Direktaufruf statt model.predict()


# --------------------------------------------------------------- Audioeingang

def _blockiert_in_echtzeit(pa, index, channels, rate, chunk):
    """Testet, ob stream.read() tatsaechlich im Audiotempo blockiert.

    Auf diesem Rechner meldet Windows dasselbe VB-Cable-Geraet ueber mehrere
    Host-APIs (MME / DirectSound / WASAPI) mit unterschiedlichen Geraete-Indizes.
    Die DirectSound-Variante hat sich im Test als kaputt herausgestellt: read()
    kehrt sofort zurueck (~650000 Aufrufe/Sekunde statt der erwarteten ~35), was
    zu verwuerfelten Buffern (vereinzelt NaN oder astronomisch grosse Werte)
    fuehrt. Deshalb wird jeder Kandidat kurz angespielt und verworfen, wenn er
    nicht ungefaehr in Echtzeit liefert.
    """
    try:
        probe = pa.open(format=FORMAT, channels=channels, rate=rate, input=True,
                         input_device_index=index, frames_per_buffer=chunk)
    except Exception:
        return False
    try:
        probe.read(chunk, exception_on_overflow=False)   # erster Read ist unzuverlaessig, verwerfen
        t0 = time.time()
        n = 5
        for _ in range(n):
            probe.read(chunk, exception_on_overflow=False)
        elapsed = time.time() - t0
        erwartet = n * chunk / rate
        return elapsed > erwartet * 0.4          # grosszuegige Toleranz nach unten
    finally:
        probe.stop_stream()
        probe.close()


def find_cable_output_device(pa):
    """Sucht das VB-Audio-Virtual-Cable-Aufnahmegeraet ueber den Namensteil 'CABLE Output'."""
    candidates = []
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        name = info["name"]
        if "CABLE Output" in name and "Point" not in name and info["maxInputChannels"] > 0:
            candidates.append((i, info))

    if not candidates:
        alle = "\n".join(
            f"  {i}: {pa.get_device_info_by_index(i)['name']} "
            f"(Eingangskanaele: {pa.get_device_info_by_index(i)['maxInputChannels']})"
            for i in range(pa.get_device_count())
        )
        raise RuntimeError(
            "FEHLER: Kein Aufnahmegeraet mit 'CABLE Output' im Namen gefunden. "
            "Ist VB-Audio Virtual Cable installiert und in Windows als Aufnahmegeraet sichtbar?\n"
            f"Vorhandene Geraete:\n{alle}"
        )

    # bevorzugt den Eintrag, der die native Rate 44100 Hz meldet (fft.py erwartet RATE=44100)
    candidates.sort(key=lambda c: 0 if c[1]["defaultSampleRate"] == RATE else 1)

    for index, info in candidates:
        channels = info["maxInputChannels"]
        if _blockiert_in_echtzeit(pa, index, channels, RATE, CHUNK):
            print(
                f"[audio] gewaehlt: Geraet {index} '{info['name']}' "
                f"({channels} Kanaele, liefert im Echtzeittempo bei {RATE} Hz)",
                flush=True,
            )
            return index, channels
        print(
            f"[audio] Geraet {index} '{info['name']}' liefert Audio NICHT im Echtzeittempo "
            f"(uebersprungen, vermutlich Treiber-/Host-API-Problem)",
            flush=True,
        )

    # Keiner der Kandidaten hat den Timing-Test bestanden -- trotzdem den besten nehmen,
    # aber deutlich warnen, damit das am Bildschirm auffaellt statt sich zu verstecken.
    index, info = candidates[0]
    channels = info["maxInputChannels"]
    print(
        f"[audio] ACHTUNG: kein 'CABLE Output'-Geraet hat den Echtzeit-Test bestanden. "
        f"Verwende trotzdem Geraet {index} '{info['name']}' -- pruefen, ob die Werte plausibel sind.",
        flush=True,
    )
    return index, channels


def audio_capture_loop(pa, device_index, device_channels):
    """Liest laufend Audio-Chunks vom CABLE-Output-Geraet und berechnet die FFT-Baender.

    Der Stream wird bewusst mit der vollen Kanalzahl des Geraets geoeffnet (VB-Cable
    meldet hier 16 Kanaele) und erst danach in numpy auf mono heruntergemischt. Ein
    direktes Oeffnen mit channels=1 auf einem 16-Kanal-DirectSound-Geraet lieferte im
    Test einen verwuerfelten Buffer (NaN-Werte) -- vermutlich ein Kanal-Handling-Bug
    des Treibers/PortAudio bei dieser Kombination.
    """
    stream = pa.open(
        format=FORMAT,
        channels=device_channels,
        rate=RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=CHUNK,
    )
    print(f"[audio] Stream offen: {RATE} Hz, {device_channels} Kanaele, Chunk {CHUNK} Samples (~{FPS} fps)", flush=True)

    n_mix = min(2, device_channels)   # Stereo-Summe (bzw. der einzige Kanal) als Mono-Signal

    try:
        while running:
            try:
                raw = stream.read(CHUNK, exception_on_overflow=False)
            except Exception as exc:
                print(f"[audio] Fehler beim Lesen: {exc}", file=sys.stderr, flush=True)
                time.sleep(0.1)
                continue

            multi = np.frombuffer(raw, dtype=np.float32).reshape(-1, device_channels)
            # Der DirectSound-Treiber liefert unter Last (GIL-Konkurrenz durch die
            # TF-Vorhersage im anderen Thread) gelegentlich vereinzelte Muell-Samples
            # im Buffer -- mal als NaN, mal als betragsmaessig astronomisch grosse,
            # aber endliche Werte (im Test bis 1e37). Ohne Saeuberung wuerde das ueber
            # FFT und Modell die komplette Vorhersage kippen. Audio-Samples liegen
            # normal in [-1, 1]; alles ausserhalb von [-8, 8] ist sicher Muell.
            multi = np.nan_to_num(multi, nan=0.0, posinf=0.0, neginf=0.0)
            multi = np.clip(multi, -8.0, 8.0)
            samples = multi[:, :n_mix].mean(axis=1)
            bins = compute_fft_bins(samples)

            with fft_lock:
                fft_queue.append(bins)
                if len(fft_queue) > 8:          # nie hinterherhinken, immer das Neueste
                    del fft_queue[:-4]
            stats["received"] += 1
            frame_available.set()
    finally:
        stream.stop_stream()
        stream.close()


# ------------------------------------------------------------------ Ausgabe

def led_output_loop():
    """Schickt fertige Vorhersagen im Original-Paketformat an ortlicht. Unveraendert."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while running:
        if not output_queue:
            if not output_ready.wait(timeout=0.5):
                continue
            output_ready.clear()
            continue
        prediction = output_queue.pop(0)

        values = np.multiply(prediction, 255).astype(np.uint8)
        for index in range(PACKETS_PER_FRAME):
            chunk = values[index * LEDS_PER_PACKET:(index + 1) * LEDS_PER_PACKET].astype(np.int16)
            chunk = np.clip(chunk - 127, -127, 127)   # Original-Offset, am oberen Rand gekappt statt gewrapped
            wire = (chunk & 0xFF).astype(np.uint8)    # Zweierkomplement fuer die Byte-Uebertragung
            header = struct.pack("!IBB", frame_count, index, 0)
            sock.sendto(header + bytes(wire.tolist()), (UDP_IP, UDP_PORT))
        stats["sent"] += 1


# --------------------------------------------------------------- Vorhersage

def prediction_loop():
    global frame_count
    last_report = time.time()
    while running:
        with fft_lock:
            batch = fft_queue.pop(0) if fft_queue else None
        if batch is None:
            if not frame_available.wait(timeout=0.5):
                continue
            frame_available.clear()
            continue

        x = np.asarray(batch, dtype=np.float32).reshape(1, INPUT_DIM)
        y = predict_fn(x)
        output_queue.append(y)
        output_ready.set()
        stats["predicted"] += 1
        frame_count += 1

        if stats["predicted"] % PRINT_EVERY == 0:
            now = time.time()
            fps = PRINT_EVERY / max(now - last_report, 1e-6)
            last_report = now
            bass = ", ".join(f"{v:.1f}" for v in x[0, :4])
            print(
                f"[lauf] empfangen {stats['received']}  vorhergesagt {stats['predicted']}  "
                f"gesendet {stats['sent']}  {fps:5.1f} fps  "
                f"fft min {x.min():.2f} max {x.max():.2f} bass[0-3] [{bass}]  "
                f"hell min {y.min():.3f} max {y.max():.3f} mittel {y.mean():.3f}",
                flush=True,
            )


# ----------------------------------------------------------------------- main

def main():
    global model, predict_fn, INPUT_DIM, running

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--target", default=UDP_IP, help="Rechner, auf dem ortlicht laeuft")
    args = parser.parse_args()

    globals()["UDP_IP"] = args.target

    if not os.path.isfile(args.model):
        print(f"FEHLER: Modell nicht gefunden: {args.model}", file=sys.stderr)
        return 2

    print(f"[modell] lade {args.model} ...", flush=True)
    import tf_keras
    model = tf_keras.models.load_model(args.model, compile=False)
    INPUT_DIM = int(model.input_shape[1])
    out_dim = int(model.output_shape[1])
    print(f"[modell] geladen: {INPUT_DIM} Eingaenge -> {out_dim} Ausgaenge", flush=True)
    if out_dim != NUM_LEDS:
        print(f"[modell] ACHTUNG: {out_dim} Ausgaenge, ortlicht erwartet {NUM_LEDS}", flush=True)
    if INPUT_DIM != len(BINS):
        print(
            f"[modell] ACHTUNG: Modell erwartet {INPUT_DIM} Werte, die FFT liefert aber "
            f"{len(BINS)} Baender. Die Vorhersage ist dann NICHT die, fuer die das Modell "
            f"trainiert wurde.",
            flush=True,
        )

    model.predict(np.zeros((1, INPUT_DIM), dtype=np.float32), verbose=0)   # warmlaufen

    # model.predict() kostet pro Einzelframe ~56 ms Verwaltung bei ~1,8 ms echter
    # Rechenzeit. Fuer Echtzeit wird das Modell deshalb direkt aufgerufen (siehe
    # main_performance.py).
    import tensorflow as tf

    @tf.function(reduce_retracing=True)
    def _graph_call(tensor):
        return model(tensor, training=False)

    def predict_fn(x):
        return _graph_call(tf.constant(x)).numpy().flatten()

    globals()["predict_fn"] = predict_fn
    for _ in range(10):
        predict_fn(np.zeros((1, INPUT_DIM), dtype=np.float32))
    print("[modell] bereit", flush=True)

    pa = pyaudio.PyAudio()
    try:
        device_index, device_channels = find_cable_output_device(pa)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        pa.terminate()
        return 3

    threading.Thread(target=led_output_loop, daemon=True).start()
    threading.Thread(target=prediction_loop, daemon=True).start()
    threading.Thread(target=audio_capture_loop, args=(pa, device_index, device_channels), daemon=True).start()
    print(f"[udp] sende an {UDP_IP}:{UDP_PORT}", flush=True)

    print("laeuft. Beenden mit Strg-C.", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        running = False
        print("\nbeendet.", flush=True)
    finally:
        pa.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
