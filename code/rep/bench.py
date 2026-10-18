"""Pruebas y benchmark de la réplica.

  python code/rep/bench.py --selftest     # SIN SNN (REP_FAKE_EVAL=1): F1-F7 -> results_selftest/informe_selftest.md
  python code/rep/bench.py --preflight    # ~30 s: ¿este Mac sirve? -> results/preflight.json
  python code/rep/bench.py --smoke        # ~10 min, SNN real, perfil smoke: S1-S10 -> results_smoke/informe_prueba.md
  python code/rep/bench.py --procs 3      # ~35 min: G0-G4, tiempos y RAM -> results/bench.json (pass_AB)

Los informes siguen una estructura fija (Entorno, Pre-vuelo, Prueba rápida, Archivos, Desviaciones, Errores, Veredicto).
Los valores (DPR, SED, cosenos) de la prueba rápida NO se juzgan: con 1 coral y L = 2 no significan nada.
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import config
from metrics9 import decide_f1_metric
from config import (DEFAULTS, PROFILES, REP, REQUIRED_PAPER_KEYS, ROOT, PAPER_REF, append_jsonl, code_hash, fake, rdir,
                    read_json, read_jsonl, write_json_atomic)

PY = sys.executable                                        # el mismo intérprete con el que se lanzó bench.py (PY de run_all.sh)
SCRIPTS = ["config", "core", "patch", "data", "engine", "ksprofile", "metrics9", "evaluate", "test_equiv", "bench",
           "run_replica", "run_ga", "run_final", "analyze"]
CLI_SCRIPTS = ["bench", "evaluate", "test_equiv", "run_replica", "run_ga", "run_final", "analyze"]   # los que tienen --help


# ============================================================================= utilidades
def run_cmd(args, env=None, timeout=None):
    """Ejecuta un comando desde neural-music/ y devuelve (código, salida combinada, segundos)."""
    e = dict(os.environ)
    e.update({k: str(v) for k, v in (env or {}).items()})
    t0 = time.time()
    try:
        p = subprocess.run([str(a) for a in args], capture_output=True, text=True, env=e, timeout=timeout, cwd=str(ROOT))
        return p.returncode, (p.stdout or "") + (p.stderr or ""), time.time() - t0
    except subprocess.TimeoutExpired:
        return 124, f"TIMEOUT tras {timeout}s", time.time() - t0


def script(name, *args):
    return [PY, str(REP / f"{name}.py"), *args]


def tail(text, n=40):
    return "\n".join(text.strip().splitlines()[-n:])


def ram_gb():
    try:
        if platform.system() == "Darwin":
            return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()) / 1e9
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                return int(line.split()[1]) / 1e6
    except Exception:
        pass
    return float("nan")


def ru_maxrss_mb():
    import resource
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v / 1e6 if platform.system() == "Darwin" else v / 1e3


def safe_clean(d):
    """Borra una carpeta de pruebas, pero nunca results/ (solo nombres que terminan en _smoke o _selftest)."""
    d = Path(d)
    if d.exists() and (d.name.endswith("_smoke") or d.name.endswith("_selftest")) and ROOT in d.resolve().parents:
        shutil.rmtree(d)


class Runner:
    def __init__(self):
        self.results, self.errors = [], []

    def run(self, tid, name, fn):
        t0 = time.time()
        try:
            ok, detail = fn()
        except Exception:
            tb = traceback.format_exc()
            ok, detail = False, "excepción: " + tb.strip().splitlines()[-1][:200]
            self.errors.append(f"[{tid}] {name}\n{tail(tb)}")
        if not ok and detail:
            self.errors.append(f"[{tid}] {name}: {detail}")
        self.results.append(dict(id=tid, name=name, ok=bool(ok), seconds=round(time.time() - t0, 1), detail=detail))
        print(f"{tid} {'PASS' if ok else 'FAIL'} · {name} ({self.results[-1]['seconds']}s) {detail}", flush=True)
        return ok

    def all_ok(self):
        return all(r["ok"] for r in self.results)


def env_md():
    v = config.versions()
    uv = run_cmd(["uv", "--version"])[1].strip() if shutil.which("uv") else "no encontrado"
    return "\n".join([
        f"- arquitectura: {platform.machine()}", f"- SO: {platform.platform()}", f"- python: {v['python']} ({sys.executable})",
        f"- uv: {uv}", *(f"- {k}: {v[k]}" for k in ("torch", "music21", "pymoo", "scipy", "numpy", "matplotlib")),
        f"- RAM: {ram_gb():.1f} GB", f"- disco libre: {shutil.disk_usage(ROOT).free / 1e9:.1f} GB", f"- code_hash: {code_hash()}",
        f"- REP_FAKE_EVAL: {os.environ.get('REP_FAKE_EVAL', '0')}", f"- REP_RESULTS_DIR: {rdir()}"])


def files_md():
    d, lines = rdir(), []
    if not d.exists():
        return "(no existe)"
    ncache = 0
    for p in sorted(d.rglob("*")):
        rel = p.relative_to(d)
        if rel.parts[0] == "cache":
            ncache += p.is_file()
            continue
        if rel.parts[0] in (".procs", "tmp_equiv") or any(x.startswith("f") and len(x) <= 3 and rel.parts[0] == x for x in rel.parts[:1]):
            continue
        if p.is_file():
            lines.append(f"{p.stat().st_size:>10,d}  {rel}")
    cache = sum(p.stat().st_size for p in (d / "cache").glob("*.json")) if (d / "cache").exists() else 0
    lines.append(f"{cache:>10,d}  cache/ ({ncache} archivos)")
    return "\n".join(lines[:120])


def write_report(fname, header_note, pre, runner, desv_default, verdict):
    d = rdir()
    desv = d / "desviaciones.md"
    desv_txt = desv.read_text().strip() if desv.exists() else desv_default
    pre_txt = "\n".join(f"- {'PASS' if ok else 'FAIL'} · {n}: {det}" for n, ok, det in pre) or "(no aplica)"
    test_txt = "\n".join(f"- {r['id']} · {'PASS' if r['ok'] else 'FAIL'} · {r['name']} · {r['seconds']} s · {r['detail']}" for r in runner.results)
    err_txt = "\n\n".join(runner.errors) if runner.errors else "(ninguno)"
    md = "\n\n".join([
        "# INFORME DE PRUEBA RÁPIDA", header_note, "## Entorno\n" + env_md(), "## Pre-vuelo\n" + pre_txt,
        "## Prueba rápida\n" + test_txt, "## Archivos\n```\n" + files_md() + "\n```", "## Desviaciones\n" + desv_txt,
        "## Errores\n" + (("```\n" + err_txt + "\n```") if runner.errors else err_txt), "## Veredicto\n" + verdict]) + "\n"
    d.mkdir(parents=True, exist_ok=True)
    (d / fname).write_text(md)
    print("\n" + "=" * 78 + "\n" + md, flush=True)
    return md


# ============================================================================= pre-vuelo
def toolchain_checks(system=None, run=None, env=None):
    """Solo macOS (Darwin): python3 del sistema (con UV_PYTHON_PREFERENCE=only-system hace falta >= 3.10) y las Xcode
    Command Line Tools (uv sync compila `dv`). En otros SO: 'no aplica'. `run` y `env` se inyectan en las pruebas."""
    system = system or platform.system()
    run = run or (lambda a: run_cmd(a)[:2])
    env = os.environ if env is None else env
    if system != "Darwin":
        return [("python3 del sistema (solo macOS)", True, "no aplica"), ("Xcode Command Line Tools (solo macOS)", True, "no aplica")]
    out = []
    code, txt = run(["python3", "--version"])
    ver = txt.strip()
    print(f"python3 --version: {ver or 'no encontrado'}", flush=True)
    import re
    m = re.search(r"(\d+)\.(\d+)", ver) if code == 0 else None
    old = (m is None) or (int(m.group(1)), int(m.group(2))) < (3, 10)
    only_system = env.get("UV_PYTHON_PREFERENCE") == "only-system"
    if old and only_system:
        out.append(("python3 del sistema ≥ 3.10 (UV_PYTHON_PREFERENCE=only-system)", False,
                    f"{ver or 'python3 no encontrado'}. Arréglalo con: \"brew install python@3.12\" o \"export UV_PYTHON_PREFERENCE=managed\""))
    else:
        out.append(("python3 del sistema ≥ 3.10 (UV_PYTHON_PREFERENCE=only-system)", True,
                    f"{ver}" + (" (< 3.10, pero UV_PYTHON_PREFERENCE no es only-system)" if old else "")))
    code, txt = run(["xcode-select", "-p"])
    out.append(("Xcode Command Line Tools (xcode-select -p)", code == 0,
                txt.strip() if code == 0 else "no instaladas. Ejecuta: \"xcode-select --install\" (hace falta para compilar dv)"))
    return out


def preflight_checks(real=True):
    """Chequeos de la sección 7.1. Devuelve [(nombre, ok, detalle)]. Los de macOS solo aplican a Darwin."""
    out = []

    def add(n, ok, d=""):
        out.append((n, bool(ok), d))
    mac = platform.system() == "Darwin"
    add("macOS en arm64", mac and platform.machine() == "arm64", f"{platform.system()} {platform.machine()}")
    r = ram_gb()
    add("RAM total ≥ 15 GB", r >= 15, f"{r:.1f} GB")
    free = shutil.disk_usage(ROOT).free / 1e9
    add("disco libre ≥ 15 GB", free >= 15, f"{free:.1f} GB")
    for tool in ("uv", "caffeinate", "zip"):
        add(f"herramienta `{tool}`", shutil.which(tool), shutil.which(tool) or "no está en PATH")
    if mac:
        _, o, _ = run_cmd(["pmset", "-g", "batt"])
        add("Mac enchufado (solo advertencia)", True, "enchufado" if "AC Power" in o else "AVISO: parece que funciona con batería")
    for n, ok, d in toolchain_checks():
        add(n, ok, d)
    v = config.versions()
    for pkg in ("torch", "music21", "scipy", "numpy", "pandas", "openpyxl", "matplotlib", "pymoo"):
        add(f"import {pkg}", v.get(pkg) is not None, v.get(pkg) or "no instalado")
    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2                      # noqa: F401
        from pymoo.core.problem import Problem                            # noqa: F401
        from pymoo.operators.crossover.sbx import SBX                     # noqa: F401
        from pymoo.operators.mutation.pm import PM                        # noqa: F401
        from pymoo.optimize import minimize                               # noqa: F401
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting   # noqa: F401
        add("imports de pymoo (5.4)", True, "ok")
    except Exception as e:
        add("imports de pymoo (5.4)", False, f"{type(e).__name__}: {e}")
    for f in ("keys.xlsx", "keyIndex.csv"):
        p = config.MUSIC / "inputs" / f
        add(f"vendor …/inputs/{f}", p.exists(), "existe" if p.exists() else "falta")
    if real:
        try:
            import core
            core.init_vendor()
            import data
            n = len(data.mxl_paths())
            add("corpus de Bach en music21: ≥ 40 .mxl", n >= 40, f"{n} archivos .mxl")
            add("los primeros 40 existen", all(Path(p).exists() for p in data.train_paths(40)), "")
        except Exception as e:
            add("corpus de Bach en music21", False, f"{type(e).__name__}: {e}")
    return out


def do_preflight():
    pre = preflight_checks(real=True)
    for n, ok, d in pre:
        print(f"{'PASS' if ok else 'FAIL'} · {n}: {d}")
    ok = all(ok for n, ok, d in pre if "solo advertencia" not in n)
    write_json_atomic(rdir() / "preflight.json", dict(ok=ok, checks=[dict(name=n, ok=o, detail=d) for n, o, d in pre],
                                                      date=time.strftime("%Y-%m-%d %H:%M:%S"), code_hash=code_hash()))
    config.manifest_add("preflight")
    print("PRE-VUELO " + ("OK" if ok else "FALLÓ: arregla lo marcado FAIL y repite"))
    return 0 if ok else 1


# ============================================================================= SELFTEST (sin SNN)
def _env_dir(sub):
    d = rdir() / sub
    d.mkdir(parents=True, exist_ok=True)
    return {"REP_RESULTS_DIR": str(d)}


class _env:
    """Cambia variables de entorno dentro de este proceso (los hijos 'spawn' las heredan)."""
    def __init__(self, **kw):
        self.kw, self.old = {k: str(v) for k, v in kw.items()}, {}

    def __enter__(self):
        for k, v in self.kw.items():
            self.old[k] = os.environ.get(k)
            os.environ[k] = v

    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def t_F1():
    import numpy as np
    from collections import Counter
    import ksprofile as K
    import metrics9 as M
    bad = []

    def chk(name, cond):
        if not cond:
            bad.append(name)
    rs = np.random.RandomState(0)
    x = rs.rand(5, 9) * 10
    chk("sed(x,x)=0", M.sed(x, x) == 0)
    chk("sed a mano = 6/1.001", abs(M.sed(np.full((2, 9), 3.0), np.array([[0.0] * 9, [2.0] * 9])) - 6 / 1.001) < 1e-9)
    ks = K.mm(K.KS_MAJOR)
    chk("mm(KS)[0]=1", ks[0] == 1.0)
    chk("mm(KS)[2]≈0.30", abs(ks[2] - 0.30) < 0.01)
    chk("mm(KS)[7]≈0.72", abs(ks[7] - 0.72) < 0.01)
    chk("cos(v,v)=1", abs(K.cos(ks, ks) - 1) < 1e-12)
    chk("cos(0,v)=0", K.cos(np.zeros(12), ks) == 0.0)
    chk("key_info", [M.key_info(k) for k in ("B- major", "f# minor", "C- major", "a minor")] ==
        [(10, "major"), (6, "minor"), (11, "major"), (9, "minor")])
    chk("oa(x,x)≈1", abs(M.oa(x[:, 0], x[:, 0]) - 1) < 0.01)
    chk("oa lejanos≈0", M.oa(x[:, 0], x[:, 0] + 100) < 0.01)
    v = [[72, 74, 76, 77, 79, 77, 76, 74], [67, 67, 69, 69, 71, 69, 67, 67], [64, 64, 64, 65, 67, 65, 64, 64],
         [48, 55, 60, 53, 55, 53, 48, 55]]
    got = dict(zip(M.NAMES, M.piece_metrics(M.piece_from_gen(v), "C major")))
    pcs = Counter(p % 12 for voice in v for p in voice)
    n = sum(pcs.values())
    pe = -sum(c / n * math.log2(c / n) for c in pcs.values())
    ints = [abs(b - a) for voice in v for a, b in zip(voice, voice[1:])]
    exp = dict(PC=14, PR=31, PI=round(sum(ints) / len(ints), 2), DPR=1.0, PE=round(pe, 2), Pp=4.0, PpR=1.0, EBR=0.0, GC=1.0)
    for k, e in exp.items():
        chk(f"pieza a mano {k}={e} (salió {got[k]})", abs(got[k] - e) < 1e-9)
    v2 = [voice[:1] + [-1] + voice[2:] for voice in v]                         # silencio en la 2.ª negra de las 4 voces
    g2 = dict(zip(M.NAMES, M.piece_metrics(M.piece_from_gen(v2), "C major")))
    chk(f"EBR con un silencio ≈ 1/8 (salió {g2['EBR']})", abs(g2["EBR"] - 0.125) < 0.006)
    # regla de f1_metric (4.4): métricas con sigma_ref < 0.01 que aportan > 50 % de sed9² -> sed_pitch5
    rs2 = np.random.RandomState(3)
    ref_c = np.column_stack([rs2.normal(m, 1.0, 10) for m in (8, 12, 2, 0.93, 4, 4)] + [np.full(10, 1.0), np.full(10, 0.0), np.full(10, 1.0)])
    ref_c[:, 3] = rs2.normal(0.93, 0.03, 10)
    gen_a = ref_c.mean(0) + np.array([0.1, 0.1, 0.05, 0.0, 0.1, -0.2, 0.04, 0.0, 0.0])         # solo difiere (0.04) una métrica casi constante
    d_a = M.decide_f1_metric([(np.tile(gen_a, (6, 1)), ref_c)] * 3)
    chk("regla f1: métricas casi constantes dominan -> sed_pitch5", d_a["f1_metric"] == "sed_pitch5" and d_a["share_low_sigma"] > 0.5)
    ref_v = rs2.normal(5, 1.0, (10, 9))
    d_b = M.decide_f1_metric([(np.tile(ref_v.mean(0) + 0.3, (6, 1)), ref_v)] * 3)
    chk("regla f1: todas con sigma > 0.01 -> sed9", d_b["f1_metric"] == "sed9" and d_b["share_low_sigma"] == 0.0)
    d_c = M.decide_f1_metric([(np.tile(ref_c.mean(0), (6, 1)), ref_c)])
    chk("regla f1: diferencias cero -> sed9, sin excepción", d_c["f1_metric"] == "sed9" and d_c["share_low_sigma"] == 0.0)
    chk("regla f1: sin datos -> sed9", M.decide_f1_metric([])["f1_metric"] == "sed9")
    chk("fitness usa el f1_metric pedido", __import__("evaluate").fitness(dict(sed9=3.0, sed_pitch5=2.0, cos_major_pasw=0.9, dpr=0.95), "sed_pitch5")[0] == 2.0
        and __import__("evaluate").fitness(dict(sed9=3.0, sed_pitch5=2.0, cos_major_pasw=0.9, dpr=0.95))[0] == 3.0)
    return (not bad), ("; ".join(bad) if bad else "sed, KS, coseno, 9 métricas de una pieza a mano, key_info, oa y regla de f1_metric correctos")


def t_F2():
    import evaluate
    base = rdir() / "f2"
    base.mkdir(parents=True, exist_ok=True)
    bad = []
    with _env(REP_RESULTS_DIR=base):
        s0 = dict(evaluate.STATS)
        r1 = evaluate.evaluate(DEFAULTS, 1, "smoke")
        r2 = evaluate.evaluate(DEFAULTS, 1, "smoke")
        if not (evaluate.STATS["computed"] == s0["computed"] + 1 and evaluate.STATS["cached"] == s0["cached"] + 1 and r1 == r2):
            bad.append("misma clave no sale de la caché")
        orig = config.code_hash
        config.code_hash = lambda: "otro-hash"
        try:
            r3 = evaluate.evaluate(DEFAULTS, 1, "smoke")
        finally:
            config.code_hash = orig
        if not (evaluate.STATS["computed"] == s0["computed"] + 2 and r3["code_hash"] == "otro-hash"):
            bad.append("cambiar code_hash no invalidó la caché")
        if len(list((base / "cache").glob("*.json"))) != 2:
            bad.append("esperaba 2 archivos en cache/")
        p = base / "x.jsonl"
        for i in range(3):
            append_jsonl(p, dict(seed=i, v=1))
        with open(p, "ab") as f:
            f.write(b'{"seed": 9, "tr')                                       # última línea truncada
        rows = read_jsonl(p)
        if len(rows) != 3 or not p.read_bytes().endswith(b"\n"):
            bad.append("JSONL truncado no se reparó")
        append_jsonl(p, dict(seed=3, v=1))
        if len(read_jsonl(p)) != 4:
            bad.append("append tras reparar")
    env = {"REP_RESULTS_DIR": str(base / "rep")}
    c, o, _ = run_cmd(script("run_replica", "--seeds", "1-3", "--profile", "smoke", "--procs", "1"), env)
    f = base / "rep" / "replica" / "C0.jsonl"
    if c != 0 or len(read_jsonl(f)) != 3:
        bad.append("run_replica 1-3 falló: " + tail(o, 5))
    else:
        lines = f.read_bytes().splitlines(keepends=True)
        f.write_bytes(b"".join(lines[:2]) + lines[2][: len(lines[2]) // 2])     # corte de luz a media línea
        c, o, _ = run_cmd(script("run_replica", "--seeds", "1-3", "--profile", "smoke", "--procs", "1"), env)
        if c != 0 or "2 semillas ya existían" not in o or len(read_jsonl(f)) != 3:
            bad.append("la reanudación tras truncar falló: " + tail(o, 5))
    return (not bad), ("; ".join(bad) if bad else "misma clave no recalcula; code_hash distinto la invalida; JSONL truncado se repara y run_replica reanuda")


def t_F3():
    import evaluate
    base = rdir() / "f3"
    base.mkdir(parents=True, exist_ok=True)
    tasks = [(DEFAULTS, 1, "smoke"), ({**DEFAULTS, "thr": 4, "izh_d": 20.0}, 2, "smoke")]
    with _env(REP_RESULTS_DIR=base):
        ctx = mp.get_context("spawn")
        with ctx.Pool(2, maxtasksperchild=1) as pool:
            res = pool.map(evaluate.eval_params, tasks, chunksize=1)
        ref = [evaluate.evaluate(p, s, pf, use_cache=False) for p, s, pf in tasks]
    ok = (len(res) == 2 and all(r["error"] is None for r in res) and res[0]["params"] != res[1]["params"]
          and all(a["sed9"] == b["sed9"] and a["gen_pitches"] == b["gen_pitches"] for a, b in zip(res, ref)))
    return ok, "2 individuos distintos evaluados en 2 procesos spawn (maxtasksperchild=1) y coinciden con la evaluación directa"


def _f1_check(gdir, resdir, want, other):
    """ga_config.json lleva f1_metric = want, pareto.json también, y la primera F de history.jsonl es r[want] (no r[other])."""
    import evaluate
    bad = []
    cfg, par = read_json(gdir / "ga_config.json"), read_json(gdir / "pareto.json")
    hist = read_jsonl(gdir / "history.jsonl", repair=False)
    if not cfg or cfg.get("f1_metric") != want:
        return [f"ga_config.f1_metric = {(cfg or {}).get('f1_metric')} (esperaba {want})"]
    if (par or {}).get("f1_metric") != want:
        bad.append(f"pareto.json sin f1_metric = {want}")
    with _env(REP_RESULTS_DIR=resdir):
        r = evaluate.evaluate(hist[0]["X"][0], cfg["s_fit"], cfg["profile"])
    f = hist[0]["F"][0][0]
    if abs(f - r[want]) > 1e-9 or abs(f - r[other]) < 1e-9:
        bad.append(f"f1 de la población inicial = {f:.6f}, esperaba {want} = {r[want]:.6f} (y no {other} = {r[other]:.6f})")
    return bad


def t_F4():
    import numpy as np
    import run_ga
    from config import encode
    bad = []
    x0 = run_ga.initial_population(1, 6)
    if not np.array_equal(x0[0], encode(DEFAULTS)):
        bad.append("fila 0 de la población inicial no es DEFAULTS")
    c, o, _ = run_cmd(script("run_ga", "--run-id", "1", "--pop", "6", "--gens", "3", "--profile", "fast", "--procs", "2"))
    g = rdir() / "ga_run1"
    if c != 0:
        return False, "run_ga falló: " + tail(o, 8)
    hist = read_jsonl(g / "history.jsonl", repair=False)
    par = read_json(g / "pareto.json")
    cfg = read_json(g / "ga_config.json")
    if not cfg:
        bad.append("falta ga_config.json")
    if len(hist) != 4:
        bad.append(f"history.jsonl tiene {len(hist)} generaciones (esperaba 4)")
    elif encode(DEFAULTS).tolist() not in [encode(p).tolist() for p in hist[0]["X"]]:
        bad.append("DEFAULTS no está en la población inicial")
    if not par or len(par.get("points", [])) < 2:
        bad.append(f"pareto.json con {len((par or {}).get('points', []))} puntos (esperaba ≥ 2)")
    b_cfg, b_hist = (g / "ga_config.json").read_bytes(), (g / "history.jsonl").read_bytes()
    c2, o2, _ = run_cmd(script("run_ga", "--run-id", "1", "--pop", "6", "--gens", "3", "--hours", "99", "--profile", "fast", "--procs", "2"))
    if c2 != 0 or "reutilizo" not in o2:
        bad.append("el relanzamiento no reutilizó ga_config.json: " + tail(o2, 5))
    elif (g / "ga_config.json").read_bytes() != b_cfg:
        bad.append("ga_config.json cambió al relanzar")
    elif (g / "history.jsonl").read_bytes() != b_hist:
        bad.append("history.jsonl no es idéntico al relanzar")
    bad += _f1_check(g, rdir(), "sed9", "sed_pitch5")                       # sin bench.json: f1 = sed9
    # --- el otro valor de f1_metric: bench.json con sed_pitch5 -> ga_config, fitness, pareto y run_final lo siguen
    sub = rdir() / "f4b"
    sub.mkdir(parents=True, exist_ok=True)
    write_json_atomic(sub / "bench.json", dict(f1_metric="sed_pitch5", selftest=True))
    e = {"REP_RESULTS_DIR": str(sub)}
    c, o, _ = run_cmd(script("run_ga", "--run-id", "1", "--pop", "6", "--gens", "2", "--profile", "fast", "--procs", "2"), e)
    gb = sub / "ga_run1"
    if c != 0:
        bad.append("run_ga con f1_metric=sed_pitch5 falló: " + tail(o, 6))
    else:
        bad += _f1_check(gb, sub, "sed_pitch5", "sed9")
        write_json_atomic(sub / "bench.json", dict(f1_metric="sed9", selftest=True))      # cambiar bench.json NO cambia una corrida ya configurada
        h0 = (gb / "history.jsonl").read_bytes()
        c, o, _ = run_cmd(script("run_ga", "--run-id", "1", "--pop", "6", "--gens", "2", "--profile", "fast", "--procs", "2"), e)
        if c != 0 or (gb / "history.jsonl").read_bytes() != h0 or read_json(gb / "ga_config.json")["f1_metric"] != "sed_pitch5":
            bad.append("al relanzar, ga_config.json no conservó f1_metric = sed_pitch5")
        c, o, _ = run_cmd(script("run_final", "--ga-run", "1", "--verify", "2", "--seeds", "1-2", "--profile", "full", "--procs", "2"), e)
        info = read_json(sub / "final" / "C1.json")
        if c != 0 or not info or info.get("f1_metric") != "sed_pitch5" or abs(info["f1"] - info["sed_pitch5"]) > 1e-12 or info["f1"] == info["sed9"]:
            bad.append("run_final no usó sed_pitch5 como f1: " + tail(o, 5))
    return (not bad), ("; ".join(bad) if bad else f"DEFAULTS = fila 0; ga_config + history (4 generaciones) + pareto ({len(par['points'])} puntos); relanzar con otro --hours reutiliza la config y reproduce history.jsonl byte a byte; f1_metric = sed9 (sin bench.json) y = sed_pitch5 (con bench.json) llegan a ga_config, fitness, pareto y run_final")


def t_F5():
    import numpy as np
    import run_final
    bad = []
    c, o, _ = run_cmd(script("run_replica", "--config", "defaults", "--seeds", "1-30", "--profile", "full", "--procs", "3"))
    if c != 0 or len(read_jsonl(rdir() / "replica" / "C0.jsonl")) != 30:
        return False, "run_replica falló: " + tail(o, 6)
    c, o, _ = run_cmd(script("run_final", "--ga-run", "1", "--verify", "3", "--seeds", "1-30", "--profile", "full", "--procs", "3"))
    fin = rdir() / "final"
    if c != 0:
        return False, "run_final falló: " + tail(o, 8)
    ver = read_jsonl(fin / "verify.jsonl", repair=False)
    c1 = read_json(fin / "C1.json")
    if len(ver) < 2 or not c1 or len(read_jsonl(fin / "C1.jsonl", repair=False)) != 30:
        bad.append("faltan verify/C1.json/C1.jsonl (30)")
    else:
        el = [r for r in ver if r["label"] != "DEFAULTS" and r["result"]["dpr"] >= 0.90]
        if el:
            d = run_final.norm_dist([[r["result"]["sed9"], 1 - r["result"]["cos_major_pasw"]] for r in el])
            if c1["label"] != el[int(np.argmin(d))]["label"]:
                bad.append(f"C1 ({c1['label']}) no es el de menor distancia al ideal ({el[int(np.argmin(d))]['label']})")
    # caso forzado: ningún individuo factible (dpr bajo) -> no se rompe y lo informa
    sub = {"REP_RESULTS_DIR": str(rdir() / "f5b"), "REP_FAKE_DPR_SHIFT": "-0.5"}
    c, o, _ = run_cmd(script("run_ga", "--run-id", "1", "--pop", "6", "--gens", "2", "--profile", "fast", "--procs", "2"), sub)
    par = read_json(rdir() / "f5b" / "ga_run1" / "pareto.json")
    if c != 0 or not par or par.get("feasible") is not False:
        bad.append("sin factibles: run_ga no dejó pareto.json con feasible=false: " + tail(o, 4))
    else:
        c, o, _ = run_cmd(script("run_final", "--ga-run", "1", "--verify", "2", "--seeds", "1-3", "--profile", "full", "--procs", "2"), sub)
        info = read_json(rdir() / "f5b" / "final" / "C1.json")
        if c != 0 or not info or "AVISO" not in o or info.get("feasible_in_full"):
            bad.append("sin factibles: run_final se rompió o no lo informó: " + tail(o, 6))
    return (not bad), ("; ".join(bad) if bad else "C0 (30) y C1 (30) corren; C1 = punto de rodilla de los verificados; sin factibles no se rompe y avisa")


def t_F6():
    bad = []
    R = rdir()
    write_json_atomic(R / "bench.json", dict(pass_AB=True, selftest=True, nota="simulado por el selftest: NO es un benchmark real"))
    c, o, _ = run_cmd(script("analyze", "--check", "--n-seeds", "30"))
    if c != 0 or "TODO COMPLETO" not in o:
        bad.append("analyze --check: " + tail(o, 8))
    c, o, _ = run_cmd(script("analyze"))
    if c != 0:
        bad.append("analyze falló: " + tail(o, 12))
    need = [f"tables/{t}.{e}" for t in ("T1", "T2", "T3", "T4", "T5") for e in ("csv", "md")] + ["tables/T1_poster.md", "tables/T3_poster.md"]
    need += [f"figs/{f}.{e}" for f in ("F1_convergencia", "F2_pareto", "F3_psc_pasw", "F4_cajas", "F5_control", "F6_metricas") for e in ("png", "pdf")]
    miss = [n for n in need if not (R / n).exists()]
    if miss:
        bad.append(f"faltan {miss}")
    # todas las diferencias cero: C1 idéntica a C0 -> p = 1, sin excepción
    z = R / "f6z"
    if z.exists():
        shutil.rmtree(z)
    (z / "final").mkdir(parents=True)
    shutil.copytree(R / "replica", z / "replica")
    shutil.copy(R / "replica" / "C0.jsonl", z / "final" / "C1.jsonl")
    shutil.copy(R / "control_hist.json", z / "control_hist.json")
    c, o, _ = run_cmd(script("analyze"), {"REP_RESULTS_DIR": str(z)})
    t3 = (z / "tables" / "T3.csv").read_text().splitlines() if (z / "tables" / "T3.csv").exists() else []
    brows = [l for l in t3 if l.startswith("(b)")]
    if c != 0 or not brows or not all(l.split(",")[-4].strip() in ("1.0", "1") for l in brows):
        bad.append("caso 'todas las diferencias cero' no dio p = 1 sin excepción: " + tail(o, 6) + " | " + " || ".join(brows[:3]))
    # T1-T3 llevan SIEMPRE sed9 y sed_pitch5; T5 trae izh_d, rama T y ε
    def txt(root, rel):
        return (root / rel).read_text() if (root / rel).exists() else ""
    for t in ("T1", "T2", "T3"):
        s = txt(R, f"tables/{t}.csv")
        if "SED (9 métricas)" not in s or "SED (5 de altura)" not in s:
            bad.append(f"{t} no trae las dos versiones del SED")
    t3 = txt(R, "tables/T3.csv")
    for grp in ("(a)", "(b)", "(c)"):
        if not all(any(l.startswith(grp) and v in l for l in t3.splitlines()) for v in ("SED (9 métricas)", "SED (5 de altura)")):
            bad.append(f"T3 {grp} sin las dos versiones del SED")
    t5 = txt(R, "tables/T5.csv")
    for frag in ("izh_d (clases)", "Rama T del vendor", "ε = 0.001"):
        if frag not in t5:
            bad.append(f"T5 sin la fila '{frag}'")
    # f1_metric = sed_pitch5 en ga_config: las figuras y el póster siguen funcionando y T1-T3 siguen llevando ambos
    y = R / "f6y"
    if y.exists():
        shutil.rmtree(y)
    shutil.copytree(R, y, ignore=shutil.ignore_patterns("f6y", "f6z", "f4b", "f5b", "f2", "f3", "f7", "cache", "tables", "figs"))
    gc = read_json(y / "ga_run1" / "ga_config.json")
    gc["f1_metric"] = "sed_pitch5"
    write_json_atomic(y / "ga_run1" / "ga_config.json", gc)
    c, o, _ = run_cmd(script("analyze"), {"REP_RESULTS_DIR": str(y)})
    if c != 0 or "f1 del AG = sed_pitch5" not in o or not (y / "figs" / "F2_pareto.png").exists():
        bad.append("analyze con f1_metric = sed_pitch5 falló: " + tail(o, 6))
    elif "SED (5 de altura)" not in txt(y, "tables/T3_poster.csv"):
        bad.append("T3_poster con f1_metric = sed_pitch5 no muestra el SED de 5 métricas")
    return (not bad), ("; ".join(bad) if bad else "--check imprime TODO COMPLETO; T1–T5 (csv+md), tablas póster y F1–F6 (png+pdf) existen; T1–T3 prueban sed9 y sed_pitch5; T5 trae izh_d, rama T y ε; con f1_metric = sed_pitch5 las figuras y el póster salen; C1 = C0 da p = 1 sin excepción")


def t_F7():
    bad = []
    c, o, _ = run_cmd(["bash", "-n", str(REP / "run_all.sh")])
    if c != 0:
        bad.append("bash -n: " + tail(o, 3))
    sub = {"REP_RESULTS_DIR": str(rdir() / "f7")}
    c, o, _ = run_cmd(["bash", str(REP / "run_all.sh"), "B"], {**sub, "DRY_RUN": "1"})
    order = [o.find(f"[ETAPA {s}]") for s in ("c0", "ga1", "final", "check", "zip")]
    if c != 0 or min(order) < 0 or order != sorted(order):
        bad.append("DRY_RUN B: etapas ausentes o desordenadas: " + tail(o, 8))
    for frag in (".done_c0", "run_replica.py --config defaults --seeds 1-30", "run_ga.py --run-id 1 --pop 12 --hours 12",
                 "run_final.py --ga-run 1 --verify 5 --seeds 1-30", "analyze.py --check", "zip -r results_bundle.zip", "-x '*.uv-cache*'"):
        if frag not in o:
            bad.append(f"falta '{frag}' en el DRY_RUN")
    if list((rdir() / "f7").glob(".done_*")):
        bad.append("el DRY_RUN creó marcas .done_")
    c, o, _ = run_cmd(["bash", str(REP / "run_all.sh"), "A"], {**sub, "DRY_RUN": "1"})
    if c != 0 or "bench.py --preflight" not in o or "bench.py --smoke" not in o:
        bad.append("DRY_RUN A: " + tail(o, 5))
    return (not bad), ("; ".join(bad) if bad else "bash -n ok; DRY_RUN B imprime c0, ga1, final, check, zip en orden con sus comandos y marcas, sin ejecutar nada; DRY_RUN A ok")


def light_preflight():
    """Pre-vuelo del selftest: sintaxis, --help de cada script e imports del entorno ligero (sin torch ni music21)."""
    out = []
    bad = []
    for n in SCRIPTS:
        try:
            compile((REP / f"{n}.py").read_text(), f"{n}.py", "exec")
        except SyntaxError as e:
            bad.append(f"{n}: {e}")
    out.append(("sintaxis de los 14 módulos", not bad, "; ".join(bad) or "ok"))
    c, o, _ = run_cmd(["bash", "-n", str(REP / "run_all.sh")])
    out.append(("bash -n run_all.sh", c == 0, tail(o, 2) or "ok"))
    bad = []
    for n in CLI_SCRIPTS:
        c, o, _ = run_cmd(script(n, "--help"), {"REP_FAKE_EVAL": "1"})
        if c != 0 or "usage" not in o.lower():
            bad.append(f"{n}: {tail(o, 2)}")
    out.append((f"--help de {len(CLI_SCRIPTS)} scripts", not bad, "; ".join(bad) or "ok"))
    v = config.versions()
    for pkg in ("numpy", "scipy", "pandas", "openpyxl", "matplotlib", "pymoo"):
        out.append((f"import {pkg} (entorno ligero)", v.get(pkg) is not None, v.get(pkg) or "no instalado"))
    out.append(("vendor, torch y music21", True, f"no se usan en el selftest (torch={v.get('torch')}, music21={v.get('music21')}, vendor presente: {config.MUSIC.exists()})"))
    ref = read_json(PAPER_REF, {})
    miss = [k for k in REQUIRED_PAPER_KEYS if k not in ref]
    out.append(("data/paper_reference.json trae todas las claves", not miss, "completo" if not miss else f"faltan {miss}"))
    # pre-vuelo de macOS (python3 / Xcode) con respuestas simuladas; en este Linux los chequeos reales salen "no aplica"
    bad = []
    lin = toolchain_checks("Linux")
    if not all(ok and d == "no aplica" for _, ok, d in lin) or len(lin) != 2:
        bad.append("Linux debería dar 'no aplica'")

    def mac(py_out, py_code, xc_code, pref="only-system"):
        run = lambda a: (py_code, py_out) if a[0] == "python3" else (xc_code, "/Library/Developer/CommandLineTools" if xc_code == 0 else "")   # noqa: E731
        return toolchain_checks("Darwin", run, {"UV_PYTHON_PREFERENCE": pref})
    r = mac("Python 3.9.6", 0, 0)
    if r[0][1] or "brew install python@3.12" not in r[0][2] or "UV_PYTHON_PREFERENCE=managed" not in r[0][2]:
        bad.append("python 3.9 + only-system debería dar FAIL con la instrucción")
    if not mac("Python 3.12.4", 0, 0)[0][1] or not mac("Python 3.9.6", 0, 0, pref="managed")[0][1]:
        bad.append("python ≥ 3.10 (o preferencia managed) debería pasar")
    r = mac("Python 3.12.4", 0, 2)
    if r[1][1] or "xcode-select --install" not in r[1][2]:
        bad.append("xcode-select -p fallido debería dar FAIL con 'xcode-select --install'")
    if mac("Python 3.12.4", 0, 0)[1][1] is not True:
        bad.append("xcode-select -p ok debería pasar")
    out.append(("pre-vuelo macOS: python3 ≥ 3.10 y Xcode CLT (respuestas simuladas)", not bad,
                "; ".join(bad) or "3.9 + only-system → FAIL con instrucción; 3.12 → PASS; sin Xcode → FAIL con instrucción; Linux → no aplica"))
    return out


def do_selftest():
    os.environ["REP_FAKE_EVAL"] = "1"
    os.environ.setdefault("REP_RESULTS_DIR", "results_selftest")
    desv = rdir() / "desviaciones.md"
    keep = desv.read_text() if desv.exists() else None           # las desviaciones se escriben a mano: sobreviven a la limpieza
    safe_clean(rdir())
    rdir().mkdir(parents=True, exist_ok=True)
    if keep:
        desv.write_text(keep)
    pre = light_preflight()
    for n, ok, d in pre:
        print(f"{'PASS' if ok else 'FAIL'} · {n}: {d}")
    R = Runner()
    for tid, name, fn in (("F1", "metrics9 y ksprofile", t_F1), ("F2", "caché, code_hash y JSONL truncado", t_F2),
                          ("F3", "Pool spawn de 2 procesos", t_F3), ("F4", "run_ga (NSGA-II, reanudación)", t_F4),
                          ("F5", "run_replica y run_final (C1 = rodilla; sin factibles)", t_F5),
                          ("F6", "analyze --check y análisis completo", t_F6), ("F7", "run_all.sh (bash -n y DRY_RUN)", t_F7)):
        R.run(tid, name, fn)
    ok = R.all_ok() and all(o for _, o, _ in pre)
    verdict = ("SELFTEST PASS" if ok else "SELFTEST FAIL") + " · SNN SIN PROBAR (se prueba en la Fase A)"
    write_report("informe_selftest.md", "Modo: selftest sin SNN (REP_FAKE_EVAL=1): evaluación simulada con el mismo esquema que la real.",
                 pre, R, "ninguna", verdict)
    return 0 if ok else 1


# ============================================================================= SMOKE (SNN real, perfil smoke)
REQUIRED_EVAL_FIELDS = ["seed", "profile", "params", "code_hash", "versions", "t_train", "t_gen", "sed9", "sed_pitch5", "oa", "dpr",
                        "cos_major_pasw", "cos_minor_pasw", "cos_major_psc", "cos_minor_psc", "psc_major", "psc_minor",
                        "pasw_major", "pasw_minor", "M_gen", "M_ref", "ref_keys", "gen_pitches", "metrics_gen_mean",
                        "metrics_ref_mean", "error"]


_VOLATILE = {"t_train", "t_gen", "params", "versions", "code_hash", "fake"}


def same_result(a, b):
    """¿Todos los campos del resultado son iguales (salvo tiempos, parámetros y metadatos)?"""
    return all(json.dumps(a.get(k), sort_keys=True) == json.dumps(b.get(k), sort_keys=True) for k in set(a) | set(b) if k not in _VOLATILE)


def fingerprint(r):
    return json.dumps([r["psc_major"], r["pasw_major"], r["psc_minor"], r["pasw_minor"], r["gen_pitches"], round(r["sed9"], 9)])


def do_smoke():
    os.environ.setdefault("REP_RESULTS_DIR", "results_smoke")
    safe_clean(rdir())
    rdir().mkdir(parents=True, exist_ok=True)
    ctx = {}
    R = Runner()
    pre = preflight_checks(real=True)
    for n, ok, d in pre:
        print(f"{'PASS' if ok else 'FAIL'} · {n}: {d}")

    def s1():
        import evaluate
        key = evaluate.cache_key(DEFAULTS, 1, "smoke", config.code_hash())
        evaluate._cache_path(key).unlink(missing_ok=True)
        r = evaluate.evaluate(DEFAULTS, 1, "smoke")
        ctx["s1"] = r
        miss = [k for k in REQUIRED_EVAL_FIELDS if k not in r]
        return (not miss and r["error"] is None), (f"faltan campos {miss}" if miss else (r["error"] or f"t_train={r['t_train']:.1f}s t_gen={r['t_gen']:.1f}s psc_major.sum={sum(r['psc_major'])}"))

    def s2():
        c, o, t = run_cmd(script("test_equiv", "--seeds", "1", "--n-train", "1", "--length", "2", "--n-gen", "1"))
        return c == 0, (tail(o, 3) if c else "gen_pitches, psc y pasw idénticos (vendor puro vs patch con DEFAULTS)")

    def s3():
        import evaluate
        base = evaluate.evaluate(DEFAULTS, 1, "smoke", use_cache=False)
        changed = {}
        for k, v in (("thr", 1), ("izh_d", 30.0), ("stdp_A", 10.0)):
            r = evaluate.evaluate({**DEFAULTS, k: v}, 1, "smoke", use_cache=False)
            changed[k] = fingerprint(r) != fingerprint(base)
            if k == "izh_d":
                changed["izh_d_algun_campo"] = not same_result(r, base)         # ¿cambió CUALQUIER campo del resultado?
        again = evaluate.evaluate(DEFAULTS, 1, "smoke", use_cache=False)
        same = fingerprint(again) == fingerprint(base) == fingerprint(ctx["s1"])
        inert = not changed.pop("izh_d_algun_campo")
        det = f"cambia el resultado: {changed}; DEFAULTS repetido idéntico: {same}"
        if inert:           # solo se reporta: quitar izh_d del AG lo decide Matías (no cuenta como FAIL)
            det += ("; **izh_d inerte: quitar del AG** (cambiar izh_d 8→30 no alteró ningún campo del resultado; "
                    "medido en perfil smoke: 1 coral, L = 2)")
        return (changed["thr"] and changed["stdp_A"] and same), det

    def s4():
        import evaluate
        t0 = time.time()
        r = evaluate.evaluate(DEFAULTS, 1, "smoke")
        dt = time.time() - t0
        return (dt < 2 and r == ctx["s1"]), f"{dt:.2f}s, idéntico al S1: {r == ctx['s1']}"

    def s5():
        import evaluate
        tasks = [(DEFAULTS, 1, "smoke"), ({**DEFAULTS, "thr": 1}, 1, "smoke")]
        with mp.get_context("spawn").Pool(2, maxtasksperchild=1) as pool:
            res = pool.map(evaluate.eval_params, tasks, chunksize=1)
        return all(r["error"] is None for r in res) and res[0]["params"] != res[1]["params"], "2 individuos en 2 procesos spawn"

    def s6():
        c, o, _ = run_cmd(script("run_ga", "--run-id", "1", "--pop", "4", "--gens", "1", "--profile", "smoke", "--procs", "2"))
        g = rdir() / "ga_run1"
        n = len(read_jsonl(g / "history.jsonl", repair=False))
        ok = c == 0 and (g / "ga_config.json").exists() and n == 2 and (g / "pareto.json").exists()
        return ok, (f"history.jsonl: {n} generaciones; pareto feasible={read_json(g / 'pareto.json', {}).get('feasible')}" if c == 0 else tail(o, 8))

    def s7():
        c1, o1, _ = run_cmd(script("run_replica", "--seeds", "1-2", "--profile", "smoke", "--procs", "2"))
        c2, o2, _ = run_cmd(script("run_final", "--ga-run", "1", "--verify", "2", "--seeds", "1-2", "--profile", "smoke", "--procs", "2"))
        n = {f: len(read_jsonl(rdir() / f, repair=False)) for f in ("replica/C0.jsonl", "final/verify.jsonl", "final/C1.jsonl")}
        ok = c1 == 0 and c2 == 0 and n["replica/C0.jsonl"] == 2 and n["final/verify.jsonl"] >= 2 and n["final/C1.jsonl"] == 2 and (rdir() / "final" / "C1.json").exists()
        return ok, (f"líneas: {n}" if ok else tail(o1 + o2, 10))

    def s8():
        c1, o1, _ = run_cmd(script("analyze", "--check", "--n-seeds", "2", "--skip-bench"))
        c2, o2, _ = run_cmd(script("analyze"))
        n_t = len(list((rdir() / "tables").glob("*.md"))) if (rdir() / "tables").exists() else 0
        n_f = len(list((rdir() / "figs").glob("*.png"))) if (rdir() / "figs").exists() else 0
        return (c1 == 0 and c2 == 0 and n_t >= 7 and n_f == 6), f"check: {'TODO COMPLETO' if c1 == 0 else tail(o1, 4)}; tablas md: {n_t}; figuras png: {n_f}" + ("" if c2 == 0 else " | " + tail(o2, 8))

    def s9():
        ref = read_json(PAPER_REF, {})
        miss = [k for k in REQUIRED_PAPER_KEYS if k not in ref]
        ch = read_json(rdir() / "control_hist.json")
        return (ch is not None and not miss), f"control_hist.json: {'sí' if ch else 'NO'}; claves del paper que faltan: {miss or 'ninguna'}"

    def s10():
        t0 = time.time()
        c1, o1, t1 = run_cmd(script("run_replica", "--seeds", "1-2", "--profile", "smoke", "--procs", "2"))
        c2, o2, t2 = run_cmd(script("run_final", "--ga-run", "1", "--verify", "2", "--seeds", "1-2", "--profile", "smoke", "--procs", "2"))
        ok = c1 == 0 and c2 == 0 and "ya existían" in o1 and "ya existían" in o2 and t1 < 5 and t2 < 5
        return ok, f"run_replica {t1:.1f}s, run_final {t2:.1f}s (deben ser < 5 s); dicen 'ya existían': {'ya existían' in o1 and 'ya existían' in o2}"

    for tid, name, fn in (("S1", "evaluate(DEFAULTS, 1, smoke) termina con todos los campos", s1),
                          ("S2", "equivalencia mini: vendor puro vs patch(DEFAULTS)", s2),
                          ("S3", "el parche engancha y no hay fuga de estado", s3), ("S4", "caché", s4),
                          ("S5", "Pool spawn de 2 procesos", s5), ("S6", "run_ga --pop 4 --gens 1", s6),
                          ("S7", "run_replica + run_final con 2 semillas", s7), ("S8", "analyze --check y analyze", s8),
                          ("S9", "control_hist.json y paper_reference.json", s9), ("S10", "reanudación sin recalcular", s10)):
        R.run(tid, name, fn)
    pre_ok = all(o for n, o, _ in pre if "solo advertencia" not in n)
    ok = R.all_ok()
    verdict = f"{'SMOKE PASS' if ok else 'SMOKE FAIL'} · LISTO PARA FASE A: {'sí' if (ok and pre_ok) else 'no'}"
    if not pre_ok:
        verdict += " (el pre-vuelo tiene FAIL)"
    write_report("informe_prueba.md", "Modo: prueba rápida con la SNN real (perfil smoke: 1 coral, L = 2). Los valores no se juzgan.",
                 pre, R, "ninguna", verdict)
    return 0 if (ok and pre_ok) else 1


# ============================================================================= BENCHMARK (G0-G4)
def _w_eval_fast(seed):
    """Worker (spawn): una evaluación fast completa con DEFAULTS, midiendo tiempo y RAM."""
    import evaluate
    t0 = time.time()
    r = evaluate.evaluate(DEFAULTS, seed, "fast", use_cache=False)
    return dict(seed=seed, wall=time.time() - t0, ram_mb=ru_maxrss_mb(), r=r)


def _w_g3():
    """Worker (spawn): t_train(40), t_gen por pieza (L=8), G1 (repetibilidad) y aceleración de fast=True con el mismo motor."""
    import data
    import engine
    import patch
    from ksprofile import psc_pasw
    t0 = time.time()
    eng, t_train = engine.build_engine(DEFAULTS, 40, 1, fast=False)
    ref = data.ref_sample(1, 1, 8)[0]
    psc0 = [v.tolist() for v in psc_pasw(eng, 0)]
    t = time.time()
    a = engine.generate(eng, ref["key"], ref["chord"], 8)
    t_a = time.time() - t
    t = time.time()
    b = engine.generate(eng, ref["key"], ref["chord"], 8)
    t_b = time.time() - t
    psc1 = [v.tolist() for v in psc_pasw(eng, 0)]
    patch.apply(DEFAULTS, fast=True)                                      # mismo motor, método rápido
    t = time.time()
    c = engine.generate(eng, ref["key"], ref["chord"], 8)
    t_c = time.time() - t
    return dict(t_train40=t_train, t_gen_L8=t_a, t_gen_L8_again=t_b, t_gen_L8_fast=t_c, g1_identical=(a == b), psc_changes=(psc0 != psc1),
                fast_identical=(a == c), speedup=(t_a / t_c if t_c > 0 else float("nan")), ram_mb=ru_maxrss_mb(), wall=time.time() - t0,
                n_skipped=getattr(eng, "n_skipped", 0))


def do_bench(procs):
    R = rdir()
    R.mkdir(parents=True, exist_ok=True)
    config.procs_guard(procs, "bench")
    config.manifest_add("bench")
    out = dict(date=time.strftime("%Y-%m-%d %H:%M:%S"), code_hash=code_hash(), procs=procs, gates={}, pass_AB=False)
    gates = out["gates"]
    ctx = mp.get_context("spawn")
    t_all = time.time()
    print("[bench] ronda A: 3 evaluaciones fast (semillas 1-3) en paralelo — G0 y G2 (≈ 18 min)", flush=True)
    with ctx.Pool(min(3, procs), maxtasksperchild=1) as pool:
        ev = pool.map(_w_eval_fast, [1, 2, 3], chunksize=1)
    r1 = ev[0]["r"]
    modes = None
    try:
        import data
        modes = data.train_modes(40)
    except Exception as e:
        modes = f"{type(e).__name__}: {e}"
    g0_ok = (r1["error"] is None and r1["dpr"] >= 0.9 and sum(r1["psc_major"]) > 0 and isinstance(modes, tuple) and modes[0] >= 8 and modes[1] >= 8)
    gates["G0"] = dict(pass_=g0_ok, **{"pass": g0_ok}, detail=dict(error=r1["error"], dpr=r1["dpr"], psc_major_sum=sum(r1["psc_major"]),
                                                                   corales_mayores_menores_en_los_40=modes, wall_s=ev[0]["wall"]))
    gates["G0"].pop("pass_")
    sed = [e["r"]["sed9"] for e in ev]
    f2 = [1 - e["r"]["cos_major_pasw"] for e in ev]
    dec = decide_f1_metric([(e["r"]["M_gen"], e["r"]["M_ref"]) for e in ev])       # regla 4.4, sin intervención humana
    out["f1_metric"] = dec["f1_metric"]
    out["f1_metric_regla"] = dec
    print(f"[bench] G2: métricas con sigma_ref < 0.01 aportan {100 * dec['share_low_sigma']:.0f} % de sed9² -> f1_metric = {dec['f1_metric']}", flush=True)
    gates["G2"] = dict(**{"pass": True}, detail=dict(sed9=sed, sed_pitch5=[e["r"]["sed_pitch5"] for e in ev], f1_metric=dec["f1_metric"],
                                                     share_low_sigma=dec["share_low_sigma"], shares_por_metrica=dec["shares_por_metrica"],
                                                     sigma_ref_media=dec["sigma_ref_media"], sd_sed9=float(__import__("numpy").std(sed)),
                                                     sd_f2=float(__import__("numpy").std(f2)),
                                                     gen_pitches_distintos=len({json.dumps(e["r"]["gen_pitches"]) for e in ev}),
                                                     nota="informativo: si casi no varían, las 30 corridas darán muchos empates"))
    t_eval_fast = float(sum(e["wall"] for e in ev) / len(ev))
    ram = max(e["ram_mb"] for e in ev)
    write_json_atomic(R / "bench.json", out)
    print("[bench] ronda B: G3 (motor de 40 corales, tiempos, G1, fast) en paralelo con G4 (equivalencia, 3 semillas)", flush=True)
    with ctx.Pool(1, maxtasksperchild=1) as pool:
        fut = pool.apply_async(_w_g3)
        c4, o4, t4 = run_cmd(script("test_equiv", "--seeds", "1-3", "--n-train", "3", "--length", "4", "--n-gen", "1"))
        g3 = fut.get()
    ram = max(ram, g3["ram_mb"])
    gates["G1"] = dict(**{"pass": bool(g3["g1_identical"])}, detail=dict(notas_identicas=g3["g1_identical"], psc_cambia_al_generar=g3["psc_changes"],
                                                                         si_falla="generar cada pieza con un motor nuevo (mucho más caro)"))
    use_fast = bool(g3["fast_identical"] and g3["speedup"] >= 1.3)
    gates["G3"] = dict(**{"pass": True}, detail=dict(t_train_5=float(ev[0]["r"]["t_train"]), t_train_40=g3["t_train40"], t_gen_pieza_L8=g3["t_gen_L8"],
                                                     t_gen_pieza_L8_fast=g3["t_gen_L8_fast"], speedup_fast=g3["speedup"],
                                                     salidas_identicas_fast=g3["fast_identical"], ram_pico_mb_por_proceso=ram,
                                                     corales_omitidos=g3["n_skipped"], regla="fast solo si acelera ≥ 1.3× y gen_pitches idéntico"))
    gates["G4"] = dict(**{"pass": c4 == 0}, detail=dict(salida=tail(o4, 6), segundos=t4))
    out["use_fast_patch"] = use_fast
    # tiempos y horas estimadas (sección 8)
    prof_f, prof_u = PROFILES["fast"], PROFILES["full"]
    t_step = g3["t_gen_L8_fast"] / 7 if use_fast else g3["t_gen_L8"] / 7
    t_eval_full = g3["t_train40"] + prof_u["n_gen"] * (prof_u["L"] - 1) * t_step
    out.update(t_eval_fast=t_eval_fast, t_eval_full=t_eval_full, t_step_s=t_step, ram_pico_mb=ram)
    pop = 12
    n_eval = math.floor(12 * 3600 * 3 / t_eval_fast)
    gens = max(5, n_eval // pop - 1)
    h = lambda secs: round(secs / 3600, 2)                                  # noqa: E731
    out["horas_estimadas"] = dict(
        ga_12h_gens_sugeridas=gens, ga=h((gens + 1) * pop * t_eval_fast / 3), verificacion_5_candidatos_y_DEFAULTS=h(math.ceil(6 / 3) * t_eval_full),
        C0_30_semillas=h(30 * t_eval_full / 3), C1_30_semillas=h(30 * t_eval_full / 3))
    out["horas_estimadas"]["total_obligatorio"] = round(sum(v for k, v in out["horas_estimadas"].items() if k != "ga_12h_gens_sugeridas"), 2)
    out["pass_AB"] = bool(gates["G0"]["pass"] and gates["G4"]["pass"])
    out["seconds"] = round(time.time() - t_all, 1)
    write_json_atomic(R / "bench.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "gates"}, indent=1))
    for g in sorted(gates):
        print(f"{g}: {'PASS' if gates[g]['pass'] else 'FAIL'}  {json.dumps(gates[g]['detail'], default=str)[:300]}")
    print("BENCH " + ("OK (pass_AB = true)" if out["pass_AB"] else "FALLÓ: pass_AB = false; la Fase B no se puede lanzar"))
    return 0 if out["pass_AB"] else 1


# ============================================================================= main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--selftest", action="store_true", help="sin SNN: F1-F7 con evaluación simulada")
    g.add_argument("--preflight", action="store_true", help="¿sirve este Mac? (~30 s)")
    g.add_argument("--smoke", action="store_true", help="prueba rápida con la SNN real, S1-S10 (~10 min)")
    ap.add_argument("--procs", type=int, default=3, help="procesos del benchmark completo G0-G4 (máx. 3)")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(do_selftest())
    if a.preflight:
        sys.exit(do_preflight())
    if a.smoke:
        sys.exit(do_smoke())
    sys.exit(do_bench(a.procs))


if __name__ == "__main__":
    main()
