#!/bin/bash
# ============================================================
#  convert_all_models.sh
#  Convierte todos los modelos del TFG a GGUF
#
#  Uso:
#    ./convert_all_models.sh
#    ./convert_all_models.sh --quants BF16,Q4_0,Q4_K_M,Q5_K_M,Q8_0
# ============================================================

MODELS_DIR="${MODELS_DIR:-$HOME/Documentos/git/TFG/Modelos}"
GGUF_OUT="${GGUF_OUT:-$HOME/Documentos/git/TFG/Modelos/GGUF}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONVERT_SCRIPT="$SCRIPT_DIR/convert_to_gguf.sh"

# Pasar args extra (--quants, --base-only, etc.) a cada conversión
EXTRA_ARGS=("$@")

# Cuantizaciones por defecto si no se pasan con --quants
DEFAULT_QUANTS="BF16,Q4_0,Q4_K_M,Q5_K_M,Q8_0"
HAS_QUANTS=false
for arg in "$@"; do [[ "$arg" == "--quants" ]] && HAS_QUANTS=true; done
if [ "$HAS_QUANTS" = false ]; then
    EXTRA_ARGS+=(--quants "$DEFAULT_QUANTS")
fi

# ── Modelos y su formato base ─────────────────────────────────
# Formato: "nombre_carpeta|base_type"
# base_type: bf16 (defecto) | fp8 | f16 | f32
declare -A MODEL_BASE_TYPE=(
    ["gemma-3n-E2B-it"]="bf16"
    ["Llama-3.2-3B-Instruct"]="bf16"
    ["Llama-3.2-1B-Instruct"]="bf16"
    ["Ministral-3-3B-Instruct-2512"]="fp8"
    ["granite-4.0-h-micro"]="bf16"
    ["DeepSeek-R1-Distill-Qwen-1.5B"]="bf16"
)

echo -e "\n\033[1m\033[36m Convirtiendo ${#MODEL_BASE_TYPE[@]} modelos → GGUF\033[0m"
echo -e "\033[2m Cuantizaciones: ${DEFAULT_QUANTS}\033[0m\n"

for MODEL in "${!MODEL_BASE_TYPE[@]}"; do
    MODEL_PATH="$MODELS_DIR/$MODEL"
    BASE_TYPE="${MODEL_BASE_TYPE[$MODEL]}"

    if [ ! -d "$MODEL_PATH" ]; then
        echo -e "\033[33m⚠  No encontrado, saltando: $MODEL\033[0m"
        continue
    fi

    echo -e "\033[1m\n════════════════════════════════════════\033[0m"
    echo -e "\033[1m $MODEL  \033[2m[base: ${BASE_TYPE^^}]\033[0m"
    echo -e "\033[1m════════════════════════════════════════\033[0m\n"

    bash "$CONVERT_SCRIPT" "$MODEL_PATH" \
        --outdir "$GGUF_OUT" \
        --base-type "$BASE_TYPE" \
        "${EXTRA_ARGS[@]}"
done

echo -e "\n\033[1m\033[32m✓ Todos los modelos procesados\033[0m"
echo -e "📁 Salida: $GGUF_OUT\n"
