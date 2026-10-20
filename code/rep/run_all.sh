#!/usr/bin/env bash
# Orquesta las fases A y B de la réplica. Uso (desde cualquier carpeta):
#   bash code/rep/run_all.sh A     # Fase A (≈ 30–40 min): uv sync, pre-vuelo, prueba rápida con la SNN y benchmark
#   nohup caffeinate -dimsu bash code/rep/run_all.sh B > logs/run_all.log 2>&1 &     # Fase B (≈ 28 h)
# Variables: PROCS (por defecto 3; usa 2 si falta RAM), GA_HOURS (por defecto 12), DRY_RUN=1 (solo imprime las etapas).
# Relanzar la misma orden reanuda: cada etapa terminada deja results/.done_<etapa> y se salta.
set -u -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT" || exit 1

# --- ÚNICO lugar donde se define el intérprete (bench.py y los scripts heredan este mismo con sys.executable)
PY="${PY:-uv run --frozen python}"

PHASE="${1:-}"
PROCS="${PROCS:-3}"
GA_HOURS="${GA_HOURS:-12}"
DRY="${DRY_RUN:-0}"
REP="code/rep"
export REP_RESULTS_DIR="${REP_RESULTS_DIR:-results}"
RES="$REP_RESULTS_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONHASHSEED=0

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "$(stamp) $*"; }

if [ "$PHASE" != "A" ] && [ "$PHASE" != "B" ]; then
  echo "Uso: bash code/rep/run_all.sh A|B   (DRY_RUN=1 para solo imprimir)"; exit 2
fi

mkdir -p logs "$RES"
# en una Terminal interactiva (Fase A) también se guarda el registro; con nohup la redirección ya lo hace
if [ -t 1 ] && [ "$DRY" != "1" ]; then exec > >(tee -a logs/run_all.log) 2>&1; fi

# env.sh: uv, Python y cachés dentro de la carpeta del proyecto
if [ -f ./env.sh ]; then source ./env.sh; fi
cd "$ROOT" || exit 1

fail() { log "FALLÓ: $*"; exit 1; }

# --- una etapa: se salta si ya existe su marca; si falla, se detiene y dice cuál
stage() {
  local name="$1"; shift
  local cmd="$*"
  if [ "$DRY" = "1" ]; then
    echo "[ETAPA $name] $cmd   (marca: $RES/.done_$name; registro: logs/$name.log)"
    return 0
  fi
  if [ -f "$RES/.done_$name" ]; then log "etapa $name: ya hecha, la salto"; return 0; fi
  log "etapa $name: inicio"
  bash -c "$cmd" 2>&1 | tee "logs/$name.log"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    log "etapa $name: FALLÓ (código $rc). No la repitas a ciegas: manda logs/ a Matías y espera."
    exit 1
  fi
  touch "$RES/.done_$name"
  log "etapa $name: fin"
}

pass_ab() {
  $PY -c "import json,sys; b=json.load(open('$RES/bench.json')); sys.exit(0 if b.get('pass_AB') else 1)" 2>/dev/null
}

if [ "$PHASE" = "A" ]; then
  log "FASE A: inicio (PROCS=$PROCS)"
  if [ "$DRY" = "1" ]; then
    echo "[A.1] uv sync"
    echo "[A.2] $PY $REP/bench.py --preflight"
    echo "[A.3] $PY $REP/bench.py --smoke"
    echo "[A.4] $PY $REP/bench.py --procs $PROCS"
    exit 0
  fi
  log "código del autor (BrainCog, commit fijado)"
  bash "$ROOT/scripts/setup_vendor.sh" 2>&1 | tee logs/setup_vendor.log
  [ "${PIPESTATUS[0]}" -eq 0 ] || fail "descarga del código del autor (mira logs/setup_vendor.log; ¿hay internet y git?)"
  if [ -x "$ROOT/.venv-vm/bin/python" ]; then
    V=$("$ROOT/.venv-vm/bin/python" -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>/dev/null)
    case "$V" in 3.10|3.11) ;; *) log "entorno anterior con Python ${V:-?} (dv necesita 3.10 o 3.11): se recrea"; rm -rf "$ROOT/.venv-vm";; esac
  fi
  log "uv sync (la primera vez descarga Python 3.11 y las dependencias)"
  uv sync 2>&1 | tee logs/uv_sync.log
  [ "${PIPESTATUS[0]}" -eq 0 ] || fail "uv sync (mira logs/uv_sync.log)"
  PYV=$($PY -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>/dev/null)
  case "$PYV" in 3.10|3.11) log "Python del entorno: $PYV";; *) fail "el entorno usa Python ${PYV:-?}; dv necesita 3.10 o 3.11 (borra .venv-vm y repite la Fase A)";; esac
  $PY "$REP/bench.py" --preflight 2>&1 | tee logs/preflight.log
  [ "${PIPESTATUS[0]}" -eq 0 ] || fail "pre-vuelo (mira results/preflight.json y logs/preflight.log)"
  $PY "$REP/bench.py" --smoke 2>&1 | tee logs/smoke.log
  [ "${PIPESTATUS[0]}" -eq 0 ] || fail "prueba rápida (manda results_smoke/informe_prueba.md y logs/)"
  $PY "$REP/bench.py" --procs "$PROCS" 2>&1 | tee logs/bench.log
  if pass_ab; then
    log "FASE A OK"
    echo "FASE A OK"
    exit 0
  fi
  G=$($PY -c "import json; b=json.load(open('$RES/bench.json')); print(', '.join(k for k,v in b.get('gates',{}).items() if not v.get('pass', True)) or '?')" 2>/dev/null)
  echo "FASE A FALLÓ: ${G:-bench.json ausente}"
  exit 1
fi

# ------------------------------------------------------------------ FASE B
log "FASE B: inicio (PROCS=$PROCS, GA_HOURS=$GA_HOURS)"
if [ "$DRY" != "1" ]; then
  pass_ab || fail "la Fase B exige results/bench.json con pass_AB = true: corre antes la Fase A (bash code/rep/run_all.sh A)"
else
  echo "(DRY_RUN: no se verifica pass_AB)"
fi

stage c0    "$PY $REP/run_replica.py --config defaults --seeds 1-30 --procs $PROCS"
stage ga1   "$PY $REP/run_ga.py --run-id 1 --pop 12 --hours $GA_HOURS --profile fast --procs $PROCS"
stage final "$PY $REP/run_final.py --ga-run 1 --verify 5 --seeds 1-30 --procs $PROCS"
stage check "$PY $REP/analyze.py --check && $PY $REP/analyze.py"

if [ "$DRY" = "1" ]; then
  echo "[ETAPA zip] zip -r results_bundle.zip $RES logs -x '*.uv-cache*'   (solo si la etapa check imprimió TODO COMPLETO; marca: $RES/.done_zip)"
  exit 0
fi
if grep -q "TODO COMPLETO" logs/check.log 2>/dev/null; then
  stage zip "zip -r results_bundle.zip $RES logs -x '*.uv-cache*'"
  log "TODO COMPLETO · results_bundle.zip listo en $(pwd)"
  echo "TODO COMPLETO"
else
  fail "la etapa check no imprimió TODO COMPLETO: no mandes nada, avisa a Matías (mira logs/check.log)"
fi
