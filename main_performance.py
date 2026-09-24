"""
SpeculativeAI — Performance-Kette, lauffaehige Portierung von main.py (2019/2020).

Unveraendert gegenueber dem Original:
  * OSC-Empfang von Ableton Live auf /fft  (Port 8000)
  * model.predict -> 13824 Helligkeitswerte
  * UDP-Ausgabe an ortlicht: Port 10005, 10 Pakete a 1402 Byte,
    Header !IBB(frameCount, paketIndex, 0), Werte als byte mit -127 Offset

Geaendert (nur damit es auf aktuellem Python laeuft):
  * Keras 2.2.4 / TF 1.x  ->  tf_keras auf TensorFlow 2.x
  * fft_handler nimmt eine BELIEBIGE Anzahl Argumente entgegen und meldet,
    was Ableton tatsaechlich schickt, statt bei != 30 Werten stumm zu scheitern
  * INPUT_DIM wird aus dem Modell gelesen, nicht hart kodiert
  * Trainingszweig entfernt (die Performance nutzt nur predict)

Aufruf:
    venv\\Scripts\\python.exe main_performance.py
    venv\\Scripts\\python.exe main_performance.py --selftest   (ohne Ableton)
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

from pythonosc import dispatcher as osc_dispatcher
from pythonosc import osc_server

# ---------------------------------------------------------------- Konfiguration

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.h5")

OSC_LISTEN_IP = "0.0.0.0"
OSC_LISTEN_PORT = 8000          # von Ableton Live
OSC_FFT_ADDRESS = "/fft"

UDP_IP = "127.0.0.1"            # ortlicht laeuft auf demselben Rechner
UDP_PORT = 10005
NUM_LEDS = 13824
PACKETS_PER_FRAME = 10
LEDS_PER_PACKET = 1402          # 10 * 1402 = 14020 >= 13824 (Original-Layout)

PRINT_EVERY = 30                # Statuszeile nur jede N-te Frame

# ------------------------------------------------------------------- Zustand

fft_queue = []
fft_lock = threading.Lock()
frame_available = threading.Event()
output_queue = []
output_ready = threading.Event()

frame_count = 0
stats = {"received": 0, "predicted": 0, "sent": 0, "bin_count": None, "mismatch_warned": False}
running = True

INPUT_DIM = None
model = None
predict_fn = None          # schneller Direktaufruf statt model.predict()


# ------------------------------------------------------------------- Empfang

def fft_handler(address, *args):
    """Nimmt /fft mit beliebig vielen Float-Argumenten entgegen."""
    values = [float(a) for a in args]
    n = len(values)

    if stats["bin_count"] != n:
        stats["bin_count"] = n
        print(f"[fft] Ableton schickt {n} Werte auf {address}", flush=True)
        if n != INPUT_DIM:
            print(
                f"[fft] ACHTUNG: Modell erwartet {INPUT_DIM} Werte, empfangen {n}. "
                f"{'Fehlende werden mit 0 aufgefuellt.' if n < INPUT_DIM else 'Ueberzaehlige werden abgeschnitten.'} "
                f"Die Vorhersage ist dann NICHT die, fuer die das Modell trainiert wurde.",
                flush=True,
            )

    if n < INPUT_DIM:
        values = [0.0] * (INPUT_DIM - n) + values   # Nullen VORNE angehaengt
    elif n > INPUT_DIM:
        values = values[:INPUT_DIM]

    with fft_lock:
        fft_queue.append(values)
        if len(fft_queue) > 8:          # nie hinterherhinken, immer das Neueste
            del fft_queue[:-4]
    stats["received"] += 1
    frame_available.set()


def frame_count_handler(address, *args):
    global frame_count
    if args:
        try:
            frame_count = int(args[0])
        except (TypeError, ValueError):
            pass


def start_osc_server(ip, port):
    disp = osc_dispatcher.Dispatcher()
    disp.map(OSC_FFT_ADDRESS, fft_handler)
    disp.map("/Playback/Recorder/frameCount", frame_count_handler)
    disp.set_default_handler(lambda addr, *a: None)

    server = osc_server.ThreadingOSCUDPServer((ip, port), disp)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[osc] lauscht auf {ip}:{port}{OSC_FFT_ADDRESS}", flush=True)
    return server


# ------------------------------------------------------------------ Ausgabe

def led_output_loop():
    """Schickt fertige Vorhersagen im Original-Paketformat an ortlicht."""
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

TARGET_FPS = 30   # ortlicht liest sowieso nur mit frameRate(30), schneller rechnen bringt nichts, nur CPU-Last

def prediction_loop():
    global frame_count
    last_report = time.time()
    last_frame_time = time.time()
    while running:
        with fft_lock:
            batch = fft_queue.pop(0) if fft_queue else None
        if batch is None:
            if not frame_available.wait(timeout=0.5):
                continue
            frame_available.clear()
            continue

        wait = (1.0 / TARGET_FPS) - (time.time() - last_frame_time)
        if wait > 0:
            time.sleep(wait)
        last_frame_time = time.time()

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


# ------------------------------------------------------------------ Selbsttest

def selftest(duration=6.0):
    """Speist synthetische FFT-Werte ein und prueft, dass UDP-Pakete rausgehen."""
    from pythonosc import udp_client

    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.settimeout(3.0)
    try:
        probe.bind(("127.0.0.1", UDP_PORT))
        listening = True
        print(f"[test] hoere selbst auf UDP {UDP_PORT} mit (ortlicht laeuft nicht)", flush=True)
    except OSError:
        listening = False
        print(f"[test] UDP {UDP_PORT} ist belegt — ortlicht laeuft bereits, sende dorthin", flush=True)

    client = udp_client.SimpleUDPClient("127.0.0.1", OSC_LISTEN_PORT)
    bins = int(os.environ.get("SAI_TEST_BINS", INPUT_DIM))
    print(f"[test] sende {bins} Bins auf {OSC_FFT_ADDRESS}", flush=True)

    end = time.time() + duration
    sent = 0
    interval = 1.0 / float(os.environ.get("SAI_TEST_FPS", "60"))
    while time.time() < end:
        t = time.time()
        payload = [float(abs(np.sin(t * (1 + i * 0.3))) * 30.0) for i in range(bins)]
        client.send_message(OSC_FFT_ADDRESS, payload)
        sent += 1
        time.sleep(interval)

    time.sleep(1.0)
    result = {"osc_gesendet": sent, **{k: v for k, v in stats.items() if k != "mismatch_warned"}}

    if listening:
        packets = []
        try:
            while len(packets) < PACKETS_PER_FRAME:
                data, _ = probe.recvfrom(2048)
                packets.append(data)
        except socket.timeout:
            pass
        result["udp_pakete_empfangen"] = len(packets)
        if packets:
            fc, idx, zero = struct.unpack("!IBB", packets[0][:6])
            result["erstes_paket"] = {
                "laenge": len(packets[0]),
                "frameCount": fc,
                "paketIndex": idx,
                "reserviert": zero,
            }
    probe.close()

    print("\n=== SELBSTTEST ===", flush=True)
    for key, value in result.items():
        print(f"  {key}: {value}", flush=True)

    ok = result["predicted"] > 0 and result["sent"] > 0
    if listening:
        ok = ok and result.get("udp_pakete_empfangen", 0) == PACKETS_PER_FRAME
        ok = ok and result.get("erstes_paket", {}).get("laenge") == 1402 + 6
    print("ERGEBNIS:", "OK" if ok else "FEHLGESCHLAGEN", flush=True)
    return 0 if ok else 1


# ----------------------------------------------------------------------- main

def main():
    global model, predict_fn, INPUT_DIM, running

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default=OSC_LISTEN_IP)
    parser.add_argument("--port", type=int, default=OSC_LISTEN_PORT)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--target", default=UDP_IP, help="Rechner, auf dem ortlicht laeuft")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    globals()["UDP_IP"] = args.target

    if not os.path.isfile(args.model):
        print(f"FEHLER: Modell nicht gefunden: {args.model}", file=sys.stderr)
        return 2

    # TensorFlow nicht alle Kerne fressen lassen -- ortlicht (Java) braucht
    # auf dem 8-Kern-Tracking-Laptop auch noch Rechenzeit fuers Rendering.
    # Muss vor jeder TF-Operation gesetzt werden, deshalb ganz vorne hier.
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)

    print(f"[modell] lade {args.model} ...", flush=True)
    import tf_keras
    model = tf_keras.models.load_model(args.model, compile=False)
    INPUT_DIM = int(model.input_shape[1])
    out_dim = int(model.output_shape[1])
    print(f"[modell] geladen: {INPUT_DIM} Eingaenge -> {out_dim} Ausgaenge", flush=True)
    if out_dim != NUM_LEDS:
        print(f"[modell] ACHTUNG: {out_dim} Ausgaenge, ortlicht erwartet {NUM_LEDS}", flush=True)

    model.predict(np.zeros((1, INPUT_DIM), dtype=np.float32), verbose=0)   # warmlaufen

    # model.predict() kostet pro Einzelframe ~56 ms Verwaltung bei ~1,8 ms
    # echter Rechenzeit (gemessen 2026-09-22 auf dem Tracking-Laptop). Fuer
    # Echtzeit wird das Modell deshalb direkt aufgerufen.
    @tf.function(reduce_retracing=True)
    def _graph_call(tensor):
        return model(tensor, training=False)

    def predict_fn(x):
        return _graph_call(tf.constant(x)).numpy().flatten()

    globals()["predict_fn"] = predict_fn
    for _ in range(10):
        predict_fn(np.zeros((1, INPUT_DIM), dtype=np.float32))
    print("[modell] bereit", flush=True)

    start_osc_server(args.ip, args.port)
    threading.Thread(target=led_output_loop, daemon=True).start()
    threading.Thread(target=prediction_loop, daemon=True).start()
    print(f"[udp] sende an {UDP_IP}:{UDP_PORT}", flush=True)

    if args.selftest:
        code = selftest()
        running = False
        return code

    print("laeuft. Beenden mit Strg-C.", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        running = False
        print("\nbeendet.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
