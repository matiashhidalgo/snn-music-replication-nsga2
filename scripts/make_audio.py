"""Convierte las piezas generadas por la SNN (campo `gen_pitches` de results/replica/C0.jsonl y results/final/C1.jsonl)
en archivos .mid y .wav para escucharlas. Compara C0 (DEFAULTS) con C1 (AG) sobre la MISMA semilla y la misma pieza de
referencia (misma tonalidad y acorde inicial).

Solo usa numpy y la biblioteca estándar (ya están en el entorno de uv): no instala nada nuevo.

  uv run --frozen python scripts/make_audio.py                     # semillas 1,2,3, pieza 0
  uv run --frozen python scripts/make_audio.py --seeds 1-5 --pieces 0,1
  REP_RESULTS_DIR=results_bundle/results uv run --frozen python scripts/make_audio.py

Salida (dentro del repo, ignorada por git): audio/C0_s01_p0.mid|wav, audio/C1_s01_p0.mid|wav y
audio/comparacion_s01_p0.wav (C0, silencio, C1). El modelo solo genera alturas: no hay duraciones, así que cada paso
dura una negra (por defecto a 80 bpm), un paso = un acorde de 4 voces; -1 = silencio.
"""
import argparse
import json
import os
import struct
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SR = 44100


def results_dir():
    p = Path(os.environ.get("REP_RESULTS_DIR", ""))
    if not str(p) or str(p) == ".":
        p = ROOT / "results" if (ROOT / "results").exists() else ROOT / "results_bundle" / "results"
    return p if p.is_absolute() else ROOT / p


def parse(s):
    out = []
    for part in str(s).split(","):
        if "-" in part:
            a, b = part.split("-")
            out += range(int(a), int(b) + 1)
        elif part.strip():
            out.append(int(part))
    return out


def load(path):
    return {r["seed"]: r for r in (json.loads(l) for l in open(path) if l.strip()) if not r.get("error")}


# ----------------------------------------------------------------------------- MIDI (formato 1, una pista por voz)
def vlq(n):
    b = [n & 0x7F]
    n >>= 7
    while n:
        b.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(b))


def write_midi(path, piece, bpm):
    ppq = 480
    tracks = []
    for v, voice in enumerate(piece):
        ev = b""
        if v == 0:
            ev += vlq(0) + b"\xff\x51\x03" + struct.pack(">I", int(60_000_000 / bpm))[1:]
        ev += vlq(0) + bytes([0xC0 | v, 19])                       # órgano de iglesia
        gap = 0
        for p in voice:
            if p is None or p < 0:
                gap += ppq
                continue
            ev += vlq(gap) + bytes([0x90 | v, int(p), 80]) + vlq(ppq - 20) + bytes([0x80 | v, int(p), 0])
            gap = 20
        ev += vlq(gap) + b"\xff\x2f\x00"
        tracks.append(b"MTrk" + struct.pack(">I", len(ev)) + ev)
    Path(path).write_bytes(b"MThd" + struct.pack(">IHHH", 6, 1, len(tracks), ppq) + b"".join(tracks))


# ----------------------------------------------------------------------------- WAV (síntesis aditiva tipo órgano)
def synth(piece, bpm):
    step = 60.0 / bpm
    n = int(SR * step * len(piece[0])) + SR // 2
    y = np.zeros(n)
    for voice in piece:
        for i, p in enumerate(voice):
            if p is None or p < 0:
                continue
            f = 440.0 * 2 ** ((p - 69) / 12)
            t = np.arange(int(SR * step * 1.02)) / SR
            w = np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * 2 * f * t) + 0.25 * np.sin(2 * np.pi * 3 * f * t)
            env = np.minimum(1, t / 0.02) * np.minimum(1, (t[-1] - t) / 0.05)
            s = int(SR * step * i)
            y[s:s + len(t)] += w * env
    return y / max(1e-9, np.abs(y).max()) * 0.8


def write_wav(path, y):
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--pieces", default="0")
    ap.add_argument("--bpm", type=float, default=80)
    a = ap.parse_args()
    R = results_dir()
    c0, c1 = load(R / "replica" / "C0.jsonl"), load(R / "final" / "C1.jsonl")
    out = ROOT / "audio"
    out.mkdir(exist_ok=True)
    gap = np.zeros(SR)
    for s in parse(a.seeds):
        for k in parse(a.pieces):
            ys = {}
            for name, d in (("C0", c0), ("C1", c1)):
                if s not in d:
                    print(f"falta la semilla {s} en {name}"); break
                piece = d[s]["gen_pitches"][k]
                stem = f"{name}_s{s:02d}_p{k}"
                write_midi(out / f"{stem}.mid", piece, a.bpm)
                ys[name] = synth(piece, a.bpm)
                write_wav(out / f"{stem}.wav", ys[name])
            if len(ys) == 2:
                write_wav(out / f"comparacion_s{s:02d}_p{k}.wav", np.concatenate([ys["C0"], gap, ys["C1"]]))
                print(f"semilla {s}, pieza {k} (tonalidad de referencia: {c0[s]['ref_keys'][k]}): listo")
    print(f"archivos en {out}")


if __name__ == "__main__":
    main()
