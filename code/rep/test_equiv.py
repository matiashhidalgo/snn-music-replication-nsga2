"""Equivalencia: vendor puro (sin patch.apply) vs patch.apply(DEFAULTS), misma semilla -> mismas notas y mismo PSC/PASW.

Los monkeypatches son globales al proceso, por eso cada variante corre en su propio subproceso (las dos a la vez).
  python code/rep/test_equiv.py --seeds 1-3          # compara (G4 de la Fase A)
  python code/rep/test_equiv.py --seeds 1 --n-train 1 --length 2   # versión mínima (S2 de la prueba rápida)
Sale con código 0 si todo es idéntico, 1 si no.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import config
from config import DEFAULTS, fake, parse_seeds, rdir, read_json, write_json_atomic


def run_variant(mode, seeds, n_train, length, n_gen, out):
    import data
    import engine
    from ksprofile import psc_pasw
    res = {}
    for s in seeds:
        eng, _ = engine.build_engine(DEFAULTS, n_train, s, vendor_only=(mode == "vendor"), fast=False)
        psc_M, pasw_M = psc_pasw(eng, 0)
        psc_m, pasw_m = psc_pasw(eng, 1)
        refs = data.ref_sample(s, n_gen, length)
        gens = [engine.generate(eng, r["key"], r["chord"], length) for r in refs]
        res[str(s)] = dict(gen_pitches=gens, psc_major=psc_M, pasw_major=pasw_M, psc_minor=psc_m, pasw_minor=pasw_m,
                           ref_keys=[r["key"] for r in refs])
    write_json_atomic(out, res)


def compare(a, b):
    rows = []
    for s in sorted(a, key=int):
        row = dict(seed=int(s))
        for k in ("gen_pitches", "psc_major", "pasw_major", "psc_minor", "pasw_minor"):
            row[k] = a[s][k] == b[s][k]               # igualdad exacta (no 'casi igual')
        row["ok"] = all(row[k] for k in ("gen_pitches", "psc_major", "pasw_major", "psc_minor", "pasw_minor"))
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="1-3")
    ap.add_argument("--n-train", type=int, default=3, help="corales de entrenamiento")
    ap.add_argument("--length", type=int, default=4, help="acordes por pieza (L)")
    ap.add_argument("--n-gen", type=int, default=1, help="piezas generadas por semilla")
    ap.add_argument("--mode", default="compare", choices=["compare", "vendor", "patch"])
    ap.add_argument("--out", default=None, help="(interno) salida de una variante")
    a = ap.parse_args()
    seeds = parse_seeds(a.seeds)
    if a.mode != "compare":
        run_variant(a.mode, seeds, a.n_train, a.length, a.n_gen, a.out)
        return
    if fake():
        print("OMITIDO: con REP_FAKE_EVAL=1 no hay vendor que comparar (esto se prueba en la Fase A).")
        return
    tmp = rdir() / "tmp_equiv"
    tmp.mkdir(parents=True, exist_ok=True)
    t0, procs, outs = time.time(), {}, {}
    for mode in ("vendor", "patch"):
        outs[mode] = tmp / f"{mode}.json"
        outs[mode].unlink(missing_ok=True)
        procs[mode] = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--mode", mode, "--seeds", a.seeds, "--n-train", str(a.n_train),
             "--length", str(a.length), "--n-gen", str(a.n_gen), "--out", str(outs[mode])])
    codes = {m: p.wait() for m, p in procs.items()}
    if any(codes.values()) or not all(o.exists() for o in outs.values()):
        print(f"ERROR: una variante falló (códigos {codes}); revisa su traceback arriba.")
        sys.exit(1)
    rows = compare(read_json(outs["vendor"]), read_json(outs["patch"]))
    ok = all(r["ok"] for r in rows)
    write_json_atomic(rdir() / "test_equiv.json", dict(ok=ok, rows=rows, seconds=round(time.time() - t0, 1),
                                                       n_train=a.n_train, length=a.length, n_gen=a.n_gen))
    for r in rows:
        print(f"semilla {r['seed']}: " + " ".join(f"{k}={'igual' if r[k] else 'DISTINTO'}" for k in
                                                   ("gen_pitches", "psc_major", "pasw_major", "psc_minor", "pasw_minor")))
    print("EQUIVALENCIA " + ("PASS" if ok else "FAIL: el patch altera el modelo"))
    config.manifest_add("test_equiv")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
