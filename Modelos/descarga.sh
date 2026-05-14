#!/bin/bash
# ============================================================
#  download_models.sh
#  Descarga modelos de HuggingFace cada uno en su carpeta
# ============================================================

set -e

# ── Configuración ────────────────────────────────────────────
BASE_DIR="${1:-./modelos}"          # Directorio base (arg1 o ./modelos)
HF_TOKEN="${HF_TOKEN:-}"           # Token desde variable de entorno

MODELS=(
    "google/gemma-3n-E2B-it"
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.2-1B-Instruct"
    "mistralai/Ministral-3-3B-Instruct-2512"
    "ibm-granite/granite-4.0-h-micro"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
)

# ── Colores ──────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ── Checks previos ───────────────────────────────────────────
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║      HuggingFace Model Downloader        ║${NC}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════╝${NC}\n"

if ! command -v hf &>/dev/null; then
    echo -e "${RED}✗ hf no encontrado.${NC}"
    echo -e "  Instala con: ${YELLOW}pip install huggingface_hub${NC}"
    exit 1
fi

if [ -z "$HF_TOKEN" ]; then
    echo -e "${YELLOW}⚠  HF_TOKEN no definido. Los modelos privados/restringidos fallarán.${NC}"
    echo -e "   Exporta tu token: ${YELLOW}export HF_TOKEN=hf_xxxxxxxxxxxx${NC}\n"
else
    echo -e "${GREEN}✓ HF_TOKEN detectado${NC}\n"
fi

mkdir -p "$BASE_DIR"
echo -e "📁 Directorio base: ${BOLD}$BASE_DIR${NC}\n"

# ── Descarga ─────────────────────────────────────────────────
TOTAL=${#MODELS[@]}
SUCCESS=0
FAILED=()

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    NUM=$((i + 1))

    # Nombre de carpeta: solo la parte después de "/"
    FOLDER_NAME=$(echo "$MODEL" | cut -d'/' -f2)
    OUT_DIR="$BASE_DIR/$FOLDER_NAME"

    echo -e "${BOLD}[$NUM/$TOTAL]${NC} ${CYAN}$MODEL${NC}"
    echo -e "        → ${OUT_DIR}"

    # Construir comando
    CMD="hf download $MODEL --local-dir $OUT_DIR"
    [ -n "$HF_TOKEN" ] && CMD="$CMD --token $HF_TOKEN"

    if $CMD; then
        echo -e "        ${GREEN}✓ Completado${NC}\n"
        SUCCESS=$((SUCCESS + 1))
    else
        echo -e "        ${RED}✗ Error descargando $MODEL${NC}\n"
        FAILED+=("$MODEL")
    fi
done

# ── Resumen ──────────────────────────────────────────────────
echo -e "${BOLD}${CYAN}══════════════════════════════════════════${NC}"
echo -e "${BOLD}Resumen:${NC} ${GREEN}$SUCCESS/$TOTAL${NC} modelos descargados"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo -e "\n${RED}Fallidos:${NC}"
    for m in "${FAILED[@]}"; do
        echo -e "  ${RED}✗${NC} $m"
    done
fi

echo -e "\n📦 Espacio usado:"
du -sh "$BASE_DIR"/*/  2>/dev/null || true
echo -e "${BOLD}${CYAN}══════════════════════════════════════════${NC}"
