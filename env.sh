# source this before running anything: keeps uv, Python and caches inside this folder
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export UV_CACHE_DIR="$ROOT/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$ROOT/.uv-python"
export UV_PROJECT_ENVIRONMENT="$ROOT/.venv-vm"
export UV_LINK_MODE=hardlink
# dv (dependencia) usa el módulo `imp`, eliminado en Python 3.12: el proyecto va fijado a 3.11 (.python-version)
# y uv lo descarga solo dentro de .uv-python; no depende del Python del sistema.
export UV_PYTHON_PREFERENCE=managed
export MPLCONFIGDIR="$ROOT/.cache/matplotlib"
export XDG_CACHE_HOME="$ROOT/.cache"
export PYTHONDONTWRITEBYTECODE=1
cd "$ROOT"
