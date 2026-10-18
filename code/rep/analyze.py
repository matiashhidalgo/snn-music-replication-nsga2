"""Análisis de resultados: SOLO lee archivos de results/ (no importa vendor, torch ni engine).

  python code/rep/analyze.py --check [--n-seeds 30] [--skip-bench]   # ¿está todo? imprime TODO COMPLETO o lo que falta
  python code/rep/analyze.py [--ga-run 1]                            # tablas T1-T5 (csv + md) y figuras F1-F6

Salidas: results/tables/T1..T5 (.csv y .md) + T1_poster.md y T3_poster.md; results/figs/F1..F6 (.png 300 dpi y .pdf).
Corre igual sobre el zip de la Fase B y sobre results_smoke/ o results_selftest/ (REP_RESULTS_DIR).
"""
import argparse
import csv
import json
import sys
import traceback

import numpy as np

import config
from config import DEFAULTS, DPR_MIN, METRICS, PAPER_REF, REQUIRED_PAPER_KEYS, SPACE, VARS, rdir, read_json, read_jsonl
from ksprofile import KS_MAJOR, KS_MINOR, cos, mm
from metrics9 import decide_f1_metric

# f1 del AG (sed9 o sed_pitch5): main() lo lee de ga_config.json; solo afecta a las figuras. Las tablas T1-T3 llevan SIEMPRE ambos.
F1K = "sed9"
SEDN = {"sed9": "SED (9 métricas)", "sed_pitch5": "SED (5 de altura)"}


# ============================================================================= carga
def load_runs(path):
    """Una corrida por semilla (la última sin error si hay repetidas), ordenadas por semilla."""
    by = {}
    for r in read_jsonl(path, repair=False):
        if r["seed"] not in by or not r.get("error") or by[r["seed"]].get("error"):
            by[r["seed"]] = r
    return [by[s] for s in sorted(by)]


def col(runs, fn):
    return np.array([fn(r) for r in runs], float)


def f2_of(r):
    return 1 - r["cos_major_pasw"]


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if isinstance(x, (bool, np.bool_)):
        return "sí" if x else "no"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return f"{x:.{nd}g}"
    return str(x)


# ============================================================================= estadística
def desc(x):
    x = np.asarray(x, float)
    return dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)), sd=float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                min=float(x.min()), max=float(x.max()))


def wilcoxon_p(d):
    """Wilcoxon de signos y rangos sobre diferencias d. Todas cero -> p = 1. Sin varianza (constante distinta de 0)
    -> NaN: no se finge significancia. Devuelve (p, nota)."""
    from scipy.stats import wilcoxon
    d = np.asarray(d, float)
    if len(d) == 0:
        return float("nan"), "sin datos"
    if np.all(d == 0):
        return 1.0, "todas las diferencias son cero"
    if np.ptp(d) == 0:
        return float("nan"), "sin varianza entre semillas"
    try:
        return float(wilcoxon(d).pvalue), ""
    except ValueError as e:
        return float("nan"), str(e)[:60]


def rank_biserial(d):
    from scipy.stats import rankdata
    d = np.asarray(d, float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = rankdata(np.abs(d))
    wp, wm = r[d > 0].sum(), r[d < 0].sum()
    return float((wp - wm) / (wp + wm))


def holm(ps):
    """Corrección de Holm-Bonferroni (NaN se respeta)."""
    ps = np.asarray(ps, float)
    out = np.full(len(ps), np.nan)
    ok = np.where(~np.isnan(ps))[0]
    order = ok[np.argsort(ps[ok])]
    m, run = len(order), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i]))
        out[i] = run
    return out


# ============================================================================= tablas
def write_table(name, header, rows, note=""):
    tdir = rdir() / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    with open(tdir / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows([[("" if (isinstance(c, float) and np.isnan(c)) else c) for c in r] for r in rows])
    md = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    md += ["| " + " | ".join(fmt(c) if not isinstance(c, str) else c for c in r) + " |" for r in rows]
    (tdir / f"{name}.md").write_text(("\n".join(md)) + ("\n\n" + note if note else "") + "\n")


def paper_quantities(ref, runs):
    """Lista de (nombre, tipo, valor del paper, vector de nuestras corridas)."""
    comp = ref["comparability"]
    q = [("cos PASW–KS mayor", "value", ref["cos_pasw_major"], col(runs, lambda r: r["cos_major_pasw"])),
         ("cos PASW–KS menor", "value", ref["cos_pasw_minor"], col(runs, lambda r: r["cos_minor_pasw"])),
         ("cos PSC–KS mayor", "value", ref["cos_psc_major"], col(runs, lambda r: r["cos_major_psc"])),
         ("cos PSC–KS menor", "value", ref["cos_psc_minor"], col(runs, lambda r: r["cos_minor_psc"])),
         ("PSC(G) mayor", "value", ref["psc_G_major"], col(runs, lambda r: mm(r["psc_major"])[7])),
         ("PASW(G) mayor", "value", ref["pasw_G_major"], col(runs, lambda r: mm(r["pasw_major"])[7]))]
    for k, name in enumerate(ref["metric_names"]):
        kind = ("value" if name in comp["value"] else "trend" if name in comp["trend"] else "ambiguous")
        q.append((name, kind, ref["metrics_model"][k], col(runs, lambda r, k=k: r["metrics_gen_mean"][k])))
    q.append((SEDN["sed9"], "magnitude", ref["sed"]["Our Model"], col(runs, lambda r: r["sed9"])))
    q.append((SEDN["sed_pitch5"], "magnitude", ref["sed"]["Our Model"], col(runs, lambda r: r["sed_pitch5"])))
    for k, name in enumerate(ref["metric_names"]):
        q.append((f"OA {name}", "info", ref["oa_model"][k], col(runs, lambda r, k=k: r["oa"][k])))
    return q


def verdict(kind, paper, x):
    d = desc(x)
    if kind == "value":
        return "coincide" if (d["min"] <= paper <= d["max"] and abs(d["median"] - paper) <= 0.05) else "no coincide"
    if kind == "magnitude":
        ratio = d["median"] / paper if paper else float("inf")
        return "mismo orden de magnitud" if 0.1 <= ratio <= 10 else "distinto orden de magnitud"
    if kind == "trend":
        return "solo tendencia (piezas completas vs L acordes)"
    if kind == "ambiguous":
        return "no comparable (definición ambigua)"
    return "informativa"


def table_T1(ref, c0):
    rows = []
    for name, kind, paper, x in paper_quantities(ref, c0):
        d = desc(x)
        p, nota = wilcoxon_p(x - paper)
        rows.append([name, kind, paper, d["n"], d["mean"], d["median"], d["sd"], d["min"], d["max"], p, verdict(kind, paper, x)])
    write_table("T1", ["cantidad", "tipo", "paper", "n", "media", "mediana", "sd", "mín", "máx", "p Wilcoxon (1 muestra)", "veredicto"],
                rows, "Veredicto 'coincide' = el valor del paper cae en [mín, máx] de las corridas y |mediana − paper| ≤ 0.05; "
                      "SED: mismo orden de magnitud. Los valores de figuras del paper valen ±0.01. Ver T5 sobre comparabilidad.")
    pick = ["cos PASW–KS mayor", "cos PASW–KS menor", "cos PSC–KS mayor", "DPR", "GC", "PpR", SEDN[F1K]]
    prow = [[r[0], fmt(r[2], 3), f"{fmt(r[5], 3)} [{fmt(r[7], 3)}–{fmt(r[8], 3)}]", r[10].split(" (")[0]] for n in pick for r in rows if r[0] == n]
    write_table("T1_poster", ["Métrica", "Paper", "C0 mediana [mín–máx]", "Veredicto"], prow)
    return rows


def table_T2(cfgs):
    rows = []
    for cname, runs in cfgs.items():
        vs = [(SEDN["sed9"], col(runs, lambda r: r["sed9"])), (SEDN["sed_pitch5"], col(runs, lambda r: r["sed_pitch5"])), ("1−cos PASW mayor (f2)", col(runs, f2_of)),
              ("DPR", col(runs, lambda r: r["dpr"])), ("OA medio", col(runs, lambda r: np.mean(r["oa"])))]
        vs += [(m, col(runs, lambda r, k=k: r["metrics_gen_mean"][k])) for k, m in enumerate(METRICS)]
        for vn, x in vs:
            d = desc(x)
            rows.append([cname, vn, d["n"], d["mean"], d["median"], d["sd"], d["min"], d["max"]])
    write_table("T2", ["config", "variable", "n", "media", "mediana", "sd", "mín", "máx"], rows)


def table_T3(ref, c0, c1):
    recs = []

    def add(group, var, a, b, paired):
        d = a - b
        p, nota = wilcoxon_p(d)
        recs.append(dict(group=group, var=var, n=len(a), A=float(np.median(a)), B=float(np.median(b)) if paired else float(b),
                         delta=float(np.median(d)), p=p, nota=nota, r=rank_biserial(d)))

    for name, kind, paper, x in paper_quantities(ref, c0):
        if kind in ("value", "magnitude"):
            add("(a) réplica C0 vs artículo", name, x, paper, False)
    if c1:
        common = sorted({r["seed"] for r in c0} & {r["seed"] for r in c1})
        a0 = [r for r in c0 if r["seed"] in common]
        a1 = [r for r in c1 if r["seed"] in common]
        for vn, fn in ((SEDN["sed9"], lambda r: r["sed9"]), (SEDN["sed_pitch5"], lambda r: r["sed_pitch5"]), ("f2 = 1−cos PASW mayor", f2_of), ("DPR", lambda r: r["dpr"])):
            add("(b) mejora C1 vs réplica C0 (pareado)", vn, col(a1, fn), col(a0, fn), True)
        for name, kind, paper, x in paper_quantities(ref, c1):
            if kind in ("value", "magnitude"):
                add("(c) mejora C1 vs artículo", name, x, paper, False)
    for g in dict.fromkeys(r["group"] for r in recs):            # Holm dentro de cada familia
        idx = [i for i, r in enumerate(recs) if r["group"] == g]
        for i, ph in zip(idx, holm([recs[i]["p"] for i in idx])):
            recs[i]["p_holm"] = float(ph)
    write_table("T3", ["comparación", "variable", "n", "mediana A", "mediana B / paper", "Δ mediana (A−B)", "p", "p Holm",
                       "rank-biserial", "nota"],
                [[r["group"], r["var"], r["n"], r["A"], r["B"], r["delta"], r["p"], r["p_holm"], r["r"], r["nota"]] for r in recs],
                "A = C0 en (a) y C1 en (b) y (c); B = C0 en (b) y el valor del paper en (a) y (c). Wilcoxon de una muestra contra el "
                "paper, pareado por semilla en (b); Holm dentro de cada familia. p vacío = sin varianza entre semillas "
                "(no se finge significancia); p = 1 si todas las diferencias son cero. Se prueban siempre las dos versiones del SED "
                "(9 métricas y 5 de altura); el valor del paper (9.24) es de 9 métricas.")
    short = {"(a)": "(a) C0 vs paper", "(b)": "(b) C1 vs C0", "(c)": "(c) C1 vs paper"}
    prow = []
    for g, v in (("(b)", SEDN[F1K]), ("(b)", "f2 = 1−cos PASW mayor"), ("(b)", "DPR"), ("(c)", SEDN[F1K]),
                 ("(c)", "cos PASW–KS mayor"), ("(c)", "DPR"), ("(a)", SEDN[F1K])):
        for r in recs:
            if r["group"].startswith(g) and r["var"] == v:
                prow.append([short[g], v, fmt(r["p_holm"], 3), fmt(r["r"], 3)])
                break
    write_table("T3_poster", ["Comparación", "Variable", "p (Holm)", "Efecto r"], prow[:7])


def table_T4(c1info):
    rows = []
    for k in VARS:
        d, lo, hi, paper = SPACE[k]
        rows.append([k, d, (c1info["params"][k] if c1info else None), "no dice" if paper is None else paper, f"[{lo}, {hi}]"])
    write_table("T4", ["parámetro", "defecto (código)", "C1 (afinada)", "paper", "cotas del AG"], rows)


def table_T5(c0, ctrl):
    sed_c = None
    if c0 and c0[0].get("M_gen") and c0[0].get("M_ref"):
        contrib = []
        for r in c0:
            g, rf = np.array(r["M_gen"]), np.array(r["M_ref"])
            contrib.append(((g.mean(0) - rf.mean(0)) / (rf.std(0) + 1e-3)) ** 2)
        contrib = np.mean(contrib, 0)
        top = np.argsort(contrib)[::-1][:3]
        sed_c = "; ".join(f"{METRICS[i]} {100 * contrib[i] / contrib.sum():.0f}%" for i in top) if contrib.sum() > 0 else "—"
    sd_sed = float(np.std(col(c0, lambda r: r["sed9"]), ddof=1)) if len(c0) > 1 else float("nan")
    dec = decide_f1_metric([(r["M_gen"], r["M_ref"]) for r in c0 if r.get("M_gen") and r.get("M_ref")])
    s9, s5 = (float(np.median(col(c0, lambda r, k=k: r[k]))) for k in ("sed9", "sed_pitch5")) if c0 else (float("nan"),) * 2
    eps_med = (f"C0: las métricas con σ_ref < 0.01 aportan {100 * dec['share_low_sigma']:.0f} % de sed9² (regla 4.4: > 50 % → sed_pitch5); "
               f"mediana sed9 = {fmt(s9, 3)} vs sed_pitch5 = {fmt(s5, 3)}")
    n_uniq = len({json.dumps(r["gen_pitches"]) for r in c0}) if c0 else 0
    cm = ctrl["major"]["cos_ks"] if ctrl else None
    cn = ctrl["minor"]["cos_ks"] if ctrl else None
    med = lambda f: float(np.median(col(c0, f))) if c0 else float("nan")        # noqa: E731
    rows = [
        ["H1", "El perfil KS NO se inyecta en las sinapsis clave→nota (comentario del autor en cortex.py:107)", "La alineación con KS tiene que surgir de las sinapsis inversas (H2)", ""],
        ["H2/H8", "PSC y PASW salen de sinapsis inversas creadas por coincidencia de disparos; PSC cuenta coincidencias", "PSC/PASW ≈ histograma de alturas del conjunto de entrenamiento",
         f"coseno histograma–KS (sin SNN): mayor {fmt(cm, 3)}, menor {fmt(cn, 3)}; SNN C0 (mediana): PASW mayor {fmt(med(lambda r: r['cos_major_pasw']), 3)}, menor {fmt(med(lambda r: r['cos_minor_pasw']), 3)}"],
        ["H3", "Umbral de creación de sinapsis: el paper pide o ≥ 5, el código usa temp ≥ 2", "Discrepancia; es hiperparámetro del AG (thr)", "paper 5, código 2"],
        ["H4", "Izhikevich: el paper dice d = 30, el código usa d = 8 (las neuronas de clave llevan d propio: 2, 4 u 8)", "Discrepancia; izh_d es hiperparámetro y no cambia las neuronas de clave", "paper 30, código 8"],
        ["H5", "STDP tipo 3: el paper no da A± ni τ±; el código usa A = 5, τ = 5 ms", "Son hiperparámetros del AG", "defecto 5 / 5"],
        ["H6", "El entrenamiento usa solo las primeras 20 notas por voz de cada coral", "La generación no puede pasar de L = 20; solo 40 de ≈408 corales", ""],
        ["H7", "La generación no tiene azar y todas las notas valen una negra; el azar está en el entrenamiento", "Poca variación entre semillas", f"sd(SED) entre semillas = {fmt(sd_sed, 3)}; piezas generadas distintas entre semillas: {n_uniq} de {len(c0)}"],
        ["Baselines", "Los parámetros de los modelos comparados (Diffusion, LSTM, …) están en el Suplemento S2", "No replicables aquí", ""],
        ["Métricas", "El paper no define PC, PR, PI, PE, Pp, PpR, EBR, GC con precisión; mide piezas completas y aquí solo los primeros L acordes", "Comparables en valor: cosenos, DPR, PpR, EBR, GC. PC, PR, PI, PE solo en tendencia", ""],
        ["GC", "Se usa GC = 1 − distancia de Hamming normalizada (el paper da GC = 1.00 en Bach)", "Con la distancia pura saldría ≈ 0", ""],
        ["Pp", "Fig. 7b da Pp = 4.19 en Bach original, imposible con 4 voces si se cuentan alturas distintas", "Aquí se cuentan alturas distintas, igual en referencia y generado", ""],
        ["SED", "σ_ref de EBR, GC y PpR suele ser ≈ 0 y ε = 0.001 amplifica cualquier diferencia", "Las métricas casi constantes pueden dominar el SED", f"contribución media al SED (C0): {sed_c or '—'}"],
        ["ε = 0.001", "Cada término de Ec. 16 es ((μ_gen − μ_ref)/(σ_ref + ε))². Con σ_ref ≈ 0 una diferencia de 0.04 (un solo silencio o unísono) pesa (0.04/0.001)² = 1600 en sed9²; con σ_ref = 0.3 pesaría (0.04/0.3)² ≈ 0.02: amplificación ×1000 en el término",
         "Por eso f1 usa sed_pitch5 si esas métricas aportan > 50 % de sed9² (G2). Se reportan y prueban siempre sed9 y sed_pitch5 (T1–T3)", eps_med],
        ["izh_d (clases)", "izh_d fija el d de Izhikevich por defecto de las clases Note, Tempo, Mode, Chord, Genre, Composer, Title, Amy, Direction y Grid. NO cambia las neuronas de clave: KeyCluster (cluster.py:143-158) las crea con d explícito (2, 4 u 8 según el perfil KS)",
         "Para izh_d solo cuentan las neuronas de nota y de modo (las que usamos para generar y para PSC/PASW); las de clave no responden. Si S3 muestra que izh_d no altera nada, es una dimensión inerte del AG", "leído en el vendor; S3 lo comprueba con la SNN real"],
        ["Rama T del vendor", "En cortex.py:468-519 (generateMelodyWithKey) la rama `durations is None` tiene la condición invertida y asigna `dic[\"N\"] = nn.selectivity` (variable `nn` del bucle anterior) donde debería ser `dic[\"T\"] = tn.selectivity`",
         "No afecta a nuestro flujo: con durations=None generateMelodyWithTone no actualiza las neuronas T, así que `tn.spiketime` queda vacío (setTestStates lo borra) y `len(tn.spiketime) > maxrate` nunca se cumple; todas las notas valen una negra (H7). Conclusión por lectura del código (no ejecutada); G4 compara contra el vendor sin parche", "no ejecutado"],
        ["Fig. 5b/5d", "Las Fig. 5b y 5d del paper son idénticas punto por punto aunque sus cosenos (Fig. 5e) difieren", "f2 usa solo el modo mayor", ""],
        ["Fig. 5c", "La Fig. 5c rotula PSC(G) = 1.0 y PASW(G) = 0.95; el texto dice 0.98 y 0.90", "Inconsistencia del paper; se usa el texto", ""],
        ["fast vs full", "El AG evalúa con 5 corales y L = 6 (fast); las corridas finales con 40 corales y L = 8 (full)", "Un proxy: por eso hay verificación en full antes de elegir C1", ""],
    ]
    write_table("T5", ["tema", "hallazgo", "impacto", "medido"], rows)


# ============================================================================= figuras
OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442", blue="#0072B2", verm="#D55E00", pink="#CC79A7")


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25, "figure.dpi": 100, "savefig.bbox": "tight"})
    return plt


def save_fig(fig, name):
    fdir = rdir() / "figs"
    fdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fdir / f"{name}.png", dpi=300)
    fig.savefig(fdir / f"{name}.pdf")
    import matplotlib.pyplot as plt
    plt.close(fig)


def hv2d(F, ref):
    """Hipervolumen 2D (minimización) respecto al punto ref."""
    P = sorted([f for f in F if f[0] < ref[0] and f[1] < ref[1]])
    hv, best2 = 0.0, ref[1]
    for f1, f2 in P:
        if f2 < best2:
            hv += (ref[0] - f1) * (best2 - f2)
            best2 = f2
    return hv


def fig_F1(hist):
    plt = _plt()
    gens = [h["gen"] for h in hist]
    feas_f1 = [min([F[0] for F, ok in zip(h["F"], h["feasible"]) if ok] or [np.nan]) for h in hist]
    all_f2 = [min([F[1] for F, ok in zip(h["F"], h["feasible"]) if ok] or [np.nan]) for h in hist]
    nfe = [h["n_feasible"] for h in hist]
    allfeas = [F for h in hist for F, ok in zip(h["F"], h["feasible"]) if ok]
    ref = (1.1 * max([f[0] for f in allfeas] or [1.0]), 1.0)
    hv = [hv2d([F for F, ok in zip(h["F"], h["feasible"]) if ok], ref) for h in hist]
    fig, ax = plt.subplots(1, 4, figsize=(11, 2.8))
    for a, y, t, c in zip(ax, (feas_f1, all_f2, nfe, hv), (f"mejor f1 ({SEDN[F1K]}), factibles", "mejor f2 (1−cos), factibles", "nº de factibles", "hipervolumen"),
                          (OI["blue"], OI["verm"], OI["green"], OI["orange"])):
        a.plot(gens, y, "-o", color=c, ms=3)
        a.set_title(t)
        a.set_xlabel("generación")
    fig.suptitle("F1 · Convergencia del NSGA-II (perfil fast)", y=1.04)
    save_fig(fig, "F1_convergencia")


def fig_F2(ver, pareto, c0, c1, ref):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.2, 4))
    if pareto and pareto.get("points"):
        P = np.array([p["F"] for p in pareto["points"]])
        ax.scatter(P[:, 0], P[:, 1], s=14, color=OI["sky"], label="frente del AG (fast)", zorder=2)
    for rec in ver:
        r = rec["result"]
        ax.scatter(r[F1K], f2_of(r), s=36, marker="s", color=OI["blue"] if rec["label"] != "DEFAULTS" else OI["black"], zorder=3)
        ax.annotate(rec["label"], (r[F1K], f2_of(r)), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.scatter([], [], marker="s", color=OI["blue"], label="candidatos verificados (full)")
    for runs, lab, c in ((c0, "C0 (30 sem.)", OI["black"]), (c1, "C1 (30 sem.)", OI["verm"])):
        if runs:
            ax.errorbar(np.mean(col(runs, lambda r: r[F1K])), np.mean(col(runs, f2_of)),
                        xerr=np.std(col(runs, lambda r: r[F1K]), ddof=1) if len(runs) > 1 else 0,
                        yerr=np.std(col(runs, f2_of), ddof=1) if len(runs) > 1 else 0, fmt="o", color=c, label=lab, capsize=2, zorder=4)
    if F1K == "sed9":                                   # el 9.24 del paper es de 9 métricas: con sed_pitch5 no se dibuja
        ax.scatter([ref["sed"]["Our Model"]], [1 - ref["cos_pasw_major"]], marker="*", s=140, color=OI["orange"], label="paper (no del todo comparable)", zorder=5)
    ax.set_xlabel(f"f1 = {SEDN[F1K]}")
    ax.set_ylabel("f2 = 1 − cos(PASW mayor, KS)")
    ax.set_title("F2 · Frente de Pareto y puntos de comparación")
    ax.legend(fontsize=7)
    save_fig(fig, "F2_pareto")


def fig_F3(c0, c1):
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    x = np.arange(12)
    for a, key, t in zip(ax, ("psc_major", "pasw_major"), ("PSC (modo mayor)", "PASW (modo mayor)")):
        a.plot(x, mm(KS_MAJOR), "k--", label="KS mayor")
        for runs, lab, c, off in ((c0, "C0", OI["blue"], -0.18), (c1, "C1", OI["verm"], 0.18)):
            if runs:
                M = np.mean([mm(r[key]) for r in runs], 0)
                S = np.std([mm(r[key]) for r in runs], 0)
                a.bar(x + off, M, width=0.34, color=c, alpha=0.8, label=lab, yerr=S, error_kw=dict(lw=0.8))
        a.set_xticks(x)
        a.set_xticklabels(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"], fontsize=7)
        a.set_title(t)
        a.set_xlabel("semitonos sobre la tónica")
    ax[0].set_ylabel("normalizado (min-max)")
    ax[0].legend(fontsize=7)
    fig.suptitle("F3 · PSC / PASW vs perfil KS (promedio de las corridas)", y=1.04)
    save_fig(fig, "F3_psc_pasw")


def fig_F4(c0, c1):
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    for a, (t, fn) in zip(ax, ((f"f1 = {SEDN[F1K]}", lambda r: r[F1K]), ("f2 = 1−cos PASW", f2_of), ("DPR", lambda r: r["dpr"]))):
        data, labs = [col(c0, fn)], ["C0"]
        if c1:
            data.append(col(c1, fn))
            labs.append("C1")
        bp = a.boxplot(data, patch_artist=True, widths=0.5)
        a.set_xticklabels(labs)
        for patch, c in zip(bp["boxes"], (OI["blue"], OI["verm"])):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        a.set_title(t)
    ax[2].axhline(DPR_MIN, color="k", ls=":", lw=1)
    fig.suptitle("F4 · Distribución en las corridas finales", y=1.04)
    save_fig(fig, "F4_cajas")


def fig_F5(ctrl, c0, ref):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    if ctrl:
        labs = ["mayor", "menor"]
        vals = {"histograma (sin SNN)": [ctrl["major"]["cos_ks"], ctrl["minor"]["cos_ks"]],
                "SNN · PSC (C0)": [np.median(col(c0, lambda r: r["cos_major_psc"])), np.median(col(c0, lambda r: r["cos_minor_psc"]))],
                "SNN · PASW (C0)": [np.median(col(c0, lambda r: r["cos_major_pasw"])), np.median(col(c0, lambda r: r["cos_minor_pasw"]))],
                "paper · PASW": [ref["cos_pasw_major"], ref["cos_pasw_minor"]]}
        w = 0.2
        for i, (k, v) in enumerate(vals.items()):
            ax.bar(np.arange(2) + (i - 1.5) * w, v, width=w, label=k, color=[OI["sky"], OI["blue"], OI["verm"], OI["orange"]][i])
        ax.set_xticks(range(2))
        ax.set_xticklabels(labs)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("coseno con el perfil KS")
        ax.legend(fontsize=7, loc="lower right")
    ax.set_title("F5 · Control: ¿alineación emergente o conteo de alturas?")
    save_fig(fig, "F5_control")


def fig_F6(c0, c1, ref):
    plt = _plt()
    fig, axs = plt.subplots(3, 3, figsize=(8, 6.4))
    for k, (a, m) in enumerate(zip(axs.ravel(), METRICS)):
        vals = [("paper orig.", ref["metrics_original"][k], OI["black"]), ("paper modelo", ref["metrics_model"][k], OI["orange"]),
                ("C0", float(np.mean(col(c0, lambda r: r["metrics_gen_mean"][k]))), OI["blue"])]
        if c1:
            vals.append(("C1", float(np.mean(col(c1, lambda r: r["metrics_gen_mean"][k]))), OI["verm"]))
        a.bar(range(len(vals)), [v[1] for v in vals], color=[v[2] for v in vals])
        a.set_xticks(range(len(vals)))
        a.set_xticklabels([v[0] for v in vals], fontsize=6, rotation=30)
        a.set_title(m)
    fig.suptitle("F6 · Las 9 métricas medias (PC, PR, PI, PE: piezas completas del paper vs L acordes aquí)", y=1.02)
    fig.tight_layout()
    save_fig(fig, "F6_metricas")


# ============================================================================= comprobación de completitud
def check(n_seeds, ga_run, skip_bench):
    R = rdir()
    items = []

    def add(name, ok, detail):
        items.append((name, ok, detail))
    c0 = [r for r in load_runs(R / "replica" / "C0.jsonl") if not r.get("error")]
    c1 = [r for r in load_runs(R / "final" / "C1.jsonl") if not r.get("error")]
    add("C0: semillas", len(c0) == n_seeds, f"{len(c0)}/{n_seeds}")
    add("C1: semillas", len(c1) == n_seeds, f"{len(c1)}/{n_seeds}")
    gc = read_json(R / f"ga_run{ga_run}" / "ga_config.json")
    hist = read_jsonl(R / f"ga_run{ga_run}" / "history.jsonl", repair=False)
    add("AG: ga_config.json", gc is not None, f"existe (f1_metric = {gc.get('f1_metric', 'sed9')})" if gc else "falta")
    if gc:
        add("AG: generaciones en history.jsonl", len(hist) == gc["gens"] + 1, f"{len(hist)}/{gc['gens'] + 1} (inicial + {gc['gens']})")
    add("AG: pareto.json", (R / f"ga_run{ga_run}" / "pareto.json").exists(), "existe" if (R / f"ga_run{ga_run}" / "pareto.json").exists() else "falta")
    nv = len(read_jsonl(R / "final" / "verify.jsonl", repair=False))
    add("final: verify.jsonl (≥ 2 líneas)", nv >= 2, f"{nv}")
    add("final: C1.json", (R / "final" / "C1.json").exists(), "existe" if (R / "final" / "C1.json").exists() else "falta")
    if not skip_bench:
        b = read_json(R / "bench.json")
        add("bench.json con pass_AB", bool(b and b.get("pass_AB")), "ok" if b and b.get("pass_AB") else "falta o pass_AB falso")
    add("control_hist.json", (R / "control_hist.json").exists(), "existe" if (R / "control_hist.json").exists() else "falta")
    add("manifest.json", (R / "manifest.json").exists(), "existe" if (R / "manifest.json").exists() else "falta")
    pr = read_json(PAPER_REF, {})
    miss = [k for k in REQUIRED_PAPER_KEYS if k not in pr]
    add("data/paper_reference.json (todas las claves)", not miss, "completo" if not miss else f"faltan {miss}")
    for n, ok, d in items:
        print(f"  [{'OK ' if ok else 'FALTA'}] {n}: {d}")
    bad = [n for n, ok, _ in items if not ok]
    if bad:
        print(f"FALTA: {', '.join(bad)}")
        return 1
    print("TODO COMPLETO")
    return 0


# ============================================================================= main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="solo verifica completitud (código ≠ 0 si falta algo)")
    ap.add_argument("--n-seeds", type=int, default=30, help="semillas esperadas por configuración (30; 2 en la prueba rápida)")
    ap.add_argument("--ga-run", type=int, default=1)
    ap.add_argument("--skip-bench", action="store_true", help="no exigir bench.json (prueba rápida / selftest)")
    a = ap.parse_args()
    if a.check:
        sys.exit(check(a.n_seeds, a.ga_run, a.skip_bench))
    R = rdir()
    ref = read_json(PAPER_REF)
    if ref is None:
        raise SystemExit(f"falta {PAPER_REF}")
    c0 = [r for r in load_runs(R / "replica" / "C0.jsonl") if not r.get("error")]
    c1 = [r for r in load_runs(R / "final" / "C1.jsonl") if not r.get("error")]
    if not c0:
        raise SystemExit(f"no hay corridas de C0 en {R / 'replica' / 'C0.jsonl'}")
    c1info = read_json(R / "final" / "C1.json")
    global F1K
    F1K = (read_json(R / f"ga_run{a.ga_run}" / "ga_config.json") or {}).get("f1_metric", "sed9")
    if F1K not in SEDN:
        F1K = "sed9"
    print(f"  f1 del AG = {F1K} (las tablas T1-T3 llevan sed9 y sed_pitch5)")
    ctrl = read_json(R / "control_hist.json")
    ver = read_jsonl(R / "final" / "verify.jsonl", repair=False)
    pareto = read_json(R / f"ga_run{a.ga_run}" / "pareto.json")
    hist = read_jsonl(R / f"ga_run{a.ga_run}" / "history.jsonl", repair=False)
    cfgs = {"C0": c0}
    if c1:
        cfgs["C1"] = c1
    for extra, path in (("C0b", R / "replica" / "C0b.jsonl"), ("C2", R / "final" / "C2.jsonl")):
        runs = [r for r in load_runs(path) if not r.get("error")]
        if runs:
            cfgs[extra] = runs
    steps = [("T1", lambda: table_T1(ref, c0)), ("T2", lambda: table_T2(cfgs)), ("T3", lambda: table_T3(ref, c0, c1)),
             ("T4", lambda: table_T4(c1info)), ("T5", lambda: table_T5(c0, ctrl)),
             ("F1", lambda: hist and fig_F1(hist)), ("F2", lambda: fig_F2(ver, pareto, c0, c1, ref)),
             ("F3", lambda: fig_F3(c0, c1)), ("F4", lambda: fig_F4(c0, c1)), ("F5", lambda: fig_F5(ctrl, c0, ref)),
             ("F6", lambda: fig_F6(c0, c1, ref))]
    failed = []
    for name, fn in steps:
        try:
            fn()
            print(f"  {name} ok")
        except Exception:
            failed.append(name)
            print(f"  {name} ERROR:\n{traceback.format_exc()}")
    config.manifest_add("analyze")
    print("analyze: " + ("listo" if not failed else f"errores en {failed}"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
