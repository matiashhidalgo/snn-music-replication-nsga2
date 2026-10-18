#!/usr/bin/env bash
# Descarga el código del autor (BrainCog) en vendor/Brain-Cog, fijado al commit exacto con el que se
# escribió y probó esta réplica. Solo baja las dos carpetas que se usan (≈ 1.5 MB). No modifica nada del vendor.
# Es idempotente: si ya está en ese commit, no hace nada. run_all.sh lo llama solo en la Fase A.
#   Uso: bash scripts/setup_vendor.sh
set -euo pipefail

SHA="f9b879f75da2247a9f0c31864a947f0e5d2f3dab"      # BrainCog-X/Brain-Cog, 2025-11-06 (Apache-2.0)
URL="https://github.com/BrainCog-X/Brain-Cog.git"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
D="$ROOT/vendor/Brain-Cog"
MUSIC="examples/Knowledge_Representation_and_Reasoning/musicMemory"

command -v git >/dev/null 2>&1 || { echo "FALLÓ: falta git. Instala las herramientas de Apple: xcode-select --install"; exit 1; }

if [ -d "$D/.git" ] && [ "$(git -C "$D" rev-parse HEAD 2>/dev/null || true)" = "$SHA" ] \
   && [ -f "$D/braincog/__init__.py" ] && [ -d "$D/$MUSIC/inputs" ]; then
  echo "vendor OK (Brain-Cog @ ${SHA:0:7})"
  exit 0
fi

echo "Descargando Brain-Cog @ ${SHA:0:7} (solo braincog/ y musicMemory/) ..."
rm -rf "$D"
mkdir -p "$D"
git -C "$D" init -q
git -C "$D" remote add origin "$URL"
git -C "$D" sparse-checkout set --cone braincog "$MUSIC"
if ! git -C "$D" fetch -q --depth 1 origin "$SHA"; then
  echo "FALLÓ: no se pudo descargar $URL (¿sin internet?). Revisa la conexión y repite el comando."
  exit 1
fi
git -C "$D" checkout -q FETCH_HEAD

[ -f "$D/braincog/__init__.py" ] && [ -d "$D/$MUSIC/inputs" ] \
  || { echo "FALLÓ: la descarga quedó incompleta. Borra vendor/ y repite el comando."; exit 1; }
echo "vendor OK (Brain-Cog @ ${SHA:0:7})"
