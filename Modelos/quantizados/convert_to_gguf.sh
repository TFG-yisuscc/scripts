#!/bin/bash
# ============================================================
#  convert_to_gguf.sh  (v2 — metadatos reales)
#  Convierte modelos HuggingFace → GGUF + Modelfile
#  usando TODOS los metadatos disponibles en los archivos
#  del modelo (config.json, tokenizer_config.json,
#  generation_config.json, special_tokens_map.json,
#  tokenizer.json, README.md, header safetensors).
#
#  Uso:
#    ./convert_to_gguf.sh <directorio_modelo> [opciones]
#
#  Opciones:
#    --outdir   DIR     Directorio de salida (default: ./gguf_output)
#    --llama    DIR     Ruta a llama.cpp      (default: ./llama.cpp)
#    --quants   LIST    Cuantizaciones por coma (default: todas)
#    --base-only        Solo F16/BF16/F32
#    --skip-convert     Saltar conversión inicial (reusar F16 existente)
# ============================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Colores ──────────────────────────────────────────────────
RED='\033[0;31m';  GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; MAGENTA='\033[0;35m'
BOLD='\033[1m';    DIM='\033[2m';      NC='\033[0m'

# ── Cuantizaciones ────────────────────────────────────────────
declare -A QUANT_DESC=(
    ["F32"]="Float 32 — máxima precisión"
    ["F16"]="Float 16 — referencia base"
    ["BF16"]="BFloat 16 — mejor rango dinámico"
    ["Q8_0"]="8-bit — casi sin pérdida"
    ["Q6_K"]="6-bit K-quant — excelente calidad"
    ["Q5_K_M"]="5-bit K-quant medio ⭐"
    ["Q5_K_S"]="5-bit K-quant pequeño"
    ["Q5_0"]="5-bit legacy"
    ["Q5_1"]="5-bit legacy v1"
    ["Q4_K_M"]="4-bit K-quant medio — uso general ⭐"
    ["Q4_K_S"]="4-bit K-quant pequeño"
    ["Q4_0"]="4-bit legacy"
    ["Q4_1"]="4-bit legacy v1"
    ["Q3_K_L"]="3-bit K-quant grande"
    ["Q3_K_M"]="3-bit K-quant medio"
    ["Q3_K_S"]="3-bit K-quant pequeño"
    ["Q2_K"]="2-bit K-quant — máxima compresión"
    ["IQ4_XS"]="4-bit iQuant extra-small"
    ["IQ4_NL"]="4-bit iQuant non-linear"
    ["IQ3_M"]="3-bit iQuant medio"
    ["IQ3_S"]="3-bit iQuant pequeño"
    ["IQ3_XS"]="3-bit iQuant extra-small"
    ["IQ2_M"]="2-bit iQuant medio"
    ["IQ2_S"]="2-bit iQuant pequeño"
    ["IQ2_XS"]="2-bit iQuant extra-small"
    ["IQ1_S"]="1-bit iQuant pequeño"
    ["IQ1_M"]="1-bit iQuant medio"
)

BASE_QUANTS=("F32" "F16" "BF16")
SECONDARY_QUANTS=(
    "Q8_0" "Q6_K"
    "Q5_K_M" "Q5_K_S" "Q5_0" "Q5_1"
    "Q4_K_M" "Q4_K_S" "Q4_0" "Q4_1"
    "Q3_K_L" "Q3_K_M" "Q3_K_S"
    "Q2_K"
    "IQ4_XS" "IQ4_NL"
    "IQ3_M" "IQ3_S" "IQ3_XS"
    "IQ2_M" "IQ2_S" "IQ2_XS"
    "IQ1_S" "IQ1_M"
)
ALL_QUANTS=("${BASE_QUANTS[@]}" "${SECONDARY_QUANTS[@]}")

# ── Defaults ─────────────────────────────────────────────────
MODEL_DIR=""
OUT_DIR="./gguf_output"
LLAMA_DIR="./llama.cpp"
SELECTED_QUANTS=("${ALL_QUANTS[@]}")
BASE_ONLY=false
SKIP_CONVERT=false
BASE_TYPE="bf16"   # Formato base por defecto

# ── Args ─────────────────────────────────────────────────────
[ $# -eq 0 ] && { echo -e "${RED}Falta directorio del modelo.${NC}\n  Uso: $0 <dir> [opciones]"; exit 1; }
MODEL_DIR="$1"; shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --outdir)       OUT_DIR="$2"; shift 2 ;;
        --llama)        LLAMA_DIR="$2"; shift 2 ;;
        --quants)       IFS=',' read -ra SELECTED_QUANTS <<< "$2"; shift 2 ;;
        --base-type)    BASE_TYPE="$2"; shift 2 ;;
        --base-only)    BASE_ONLY=true; shift ;;
        --skip-convert) SKIP_CONVERT=true; shift ;;
        *) echo -e "${RED}Opción desconocida: $1${NC}"; exit 1 ;;
    esac
done

# ── Helpers ───────────────────────────────────────────────────
log()     { echo -e "${CYAN}▶${NC} $*"; }
ok()      { echo -e "${GREEN}✓${NC} $*"; }
warn()    { echo -e "${YELLOW}⚠${NC} $*"; }
err()     { echo -e "${RED}✗${NC} $*"; }
section() { echo -e "\n${BOLD}${MAGENTA}━━━ $* ━━━${NC}"; }

is_base_quant() {
    local q="$1"
    for bq in "${BASE_QUANTS[@]}"; do [[ "$q" == "$bq" ]] && return 0; done
    return 1
}

# Leer un campo del JSON de metadatos
jget() {
    echo "${META_JSON}" | python3 -c \
        "import json,sys; d=json.load(sys.stdin); v=d.get('$1',''); print(v if v else '')" \
        2>/dev/null || true
}
jget_list() {
    echo "${META_JSON}" | python3 -c "
import json, sys
d = json.load(sys.stdin)
val = d.get('$1', [])
if isinstance(val, list):
    for x in val: print(x)
elif val:
    print(val)
" 2>/dev/null || true
}

# ── Header ───────────────────────────────────────────────────
echo -e "${BOLD}${CYAN}"
echo "╔══════════════════════════════════════════════════╗"
echo "║    SafeTensors → GGUF  (metadatos completos)     ║"
echo "╚══════════════════════════════════════════════════╝"
echo -e "${NC}"

# ── Validaciones ─────────────────────────────────────────────
section "Validando entorno"
[ ! -d "$MODEL_DIR" ] && { err "Directorio no existe: $MODEL_DIR"; exit 1; }
ok "Modelo: $MODEL_DIR"
MODEL_NAME=$(basename "$MODEL_DIR")

# ── Extraer metadatos reales ──────────────────────────────────
section "Extrayendo metadatos"

EXTRACT_SCRIPT="$SCRIPT_DIR/extract_metadata.py"
if [ ! -f "$EXTRACT_SCRIPT" ]; then
    err "No se encuentra extract_metadata.py en $SCRIPT_DIR"
    exit 1
fi

META_JSON=$(python3 "$EXTRACT_SCRIPT" "$MODEL_DIR" 2>/dev/null || echo '{}')

# Mostrar fuentes encontradas
echo "${META_JSON}" | python3 -c "
import json, sys
d = json.load(sys.stdin)
sources = d.get('_sources', {})
for k, v in sources.items():
    icon = '✓' if v else '·'
    color = '\033[32m' if v else '\033[2m'
    print(f'  {color}{icon}\033[0m {k}')
" 2>/dev/null || true

# ── Leer valores extraídos ────────────────────────────────────
ARCH=$(jget "architecture")
CONTEXT_LEN=$(jget "context_length"); CONTEXT_LEN="${CONTEXT_LEN:-4096}"
LICENSE=$(jget "license")
DESCRIPTION=$(jget "description")
BASE_MODEL=$(jget "base_model")
AUTHOR=$(jget "author")
VERSION=$(jget "version")
PIPELINE_TAG=$(jget "pipeline_tag")
TRANSFORMERS_VERSION=$(jget "transformers_version")
BOS_TOKEN=$(jget "bos_token")
EOS_TOKEN=$(jget "eos_token")
CHAT_TEMPLATE=$(jget "chat_template")
TOKENIZER_CLASS=$(jget "tokenizer_class")
TEMPERATURE=$(jget "temperature")
TOP_P=$(jget "top_p")
TOP_K=$(jget "top_k")
REPEAT_PENALTY=$(jget "repeat_penalty")
NUM_PREDICT=$(jget "num_predict")
LAYER_COUNT=$(jget "layer_count")
VOCAB_SIZE=$(jget "vocab_size")
EMBEDDING_LEN=$(jget "embedding_length")
HEAD_COUNT=$(jget "head_count")
mapfile -t STOP_TOKENS < <(jget_list "stop_tokens")
mapfile -t LANGUAGES   < <(jget_list "languages")

echo ""
[ -n "$ARCH" ]        && log "Arquitectura : $ARCH"
[ -n "$CONTEXT_LEN" ] && log "Contexto     : $CONTEXT_LEN tokens"
[ -n "$LAYER_COUNT" ] && log "Capas        : $LAYER_COUNT"
[ -n "$VOCAB_SIZE" ]  && log "Vocabulario  : $VOCAB_SIZE tokens"
[ -n "$LICENSE" ]     && log "Licencia     : $LICENSE"
[ -n "$AUTHOR" ]      && log "Autor        : $AUTHOR"
[ -n "$BASE_MODEL" ]  && log "Base model   : $BASE_MODEL"
[ -n "$EOS_TOKEN" ]   && log "EOS token    : $EOS_TOKEN"
[ ${#STOP_TOKENS[@]} -gt 0 ] && log "Stop tokens  : ${STOP_TOKENS[*]}"
[ -n "$TEMPERATURE" ] && log "Temperature  : $TEMPERATURE"
[ -n "$TOP_P" ]       && log "Top-p        : $TOP_P"
[ -n "$CHAT_TEMPLATE" ] && log "Chat template: encontrado ($(echo "$CHAT_TEMPLATE" | wc -c) chars)"

# ── Validar llama.cpp ─────────────────────────────────────────
CONVERT_SCRIPT="$LLAMA_DIR/convert_hf_to_gguf.py"
QUANTIZE_BIN="$LLAMA_DIR/build/bin/llama-quantize"

if [ ! -f "$CONVERT_SCRIPT" ] && [ "$SKIP_CONVERT" = false ]; then
    warn "llama.cpp no encontrado en $LLAMA_DIR. Clonando..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp "$LLAMA_DIR"
    pip install -r "$LLAMA_DIR/requirements.txt" -q
fi

if [ ! -f "$QUANTIZE_BIN" ] && [ "$BASE_ONLY" = false ]; then
    warn "llama-quantize no compilado. Compilando..."
    (cd "$LLAMA_DIR" && cmake -B build -DGGML_NATIVE=OFF 2>/dev/null && \
     cmake --build build --config Release -j"$(nproc)" 2>/dev/null)
    ok "llama.cpp compilado"
fi

# ── Preparar salida ───────────────────────────────────────────
section "Preparando salida"
MODEL_OUT_DIR="$OUT_DIR/$MODEL_NAME"
mkdir -p "$MODEL_OUT_DIR"
log "Directorio: $MODEL_OUT_DIR"

# JSON de metadatos para pasar al conversor
GGUF_META_JSON=$(echo "$META_JSON" | python3 -c "
import json, sys
meta = json.load(sys.stdin)
out = {}
def add(k, v):
    if v and str(v).strip(): out[k] = v

add('general.architecture', meta.get('architecture'))
add('general.author',       meta.get('author'))
add('general.version',      meta.get('version'))
add('general.description',  meta.get('description'))
add('general.license',      meta.get('license'))
add('general.source.url',   meta.get('source_path') or meta.get('base_model'))
add('general.base_model',   meta.get('base_model'))
add('general.pipeline_tag', meta.get('pipeline_tag'))
langs = meta.get('languages', [])
if langs: add('general.languages', ','.join(langs))
print(json.dumps(out))
" 2>/dev/null || echo '{}')

# ── Función: generar Modelfile ────────────────────────────────
generate_modelfile() {
    local gguf_path="$1"
    local quant="$2"
    local mf_path="${gguf_path%.gguf}.Modelfile"

    python3 - "$gguf_path" "$quant" "$MODEL_NAME" "$META_JSON" \
        "${QUANT_DESC[$quant]:-}" "$mf_path" \
        "${STOP_TOKENS[@]+"${STOP_TOKENS[@]}"}" <<'PYEOF'
import sys, json, os, textwrap

gguf_path  = sys.argv[1]
quant      = sys.argv[2]
model_name = sys.argv[3]
meta_json  = sys.argv[4]
quant_desc = sys.argv[5]
mf_path    = sys.argv[6]
stop_tokens = [s for s in sys.argv[7:] if s.strip()]

try:
    meta = json.loads(meta_json)
except Exception:
    meta = {}

arch         = meta.get("architecture", "")
context_len  = str(meta.get("context_length", "4096"))
license_     = meta.get("license", "")
description  = meta.get("description", "")
base_model   = meta.get("base_model", "")
author       = meta.get("author", "")
version      = meta.get("version", "")
languages    = meta.get("languages", [])
bos_token    = meta.get("bos_token", "")
eos_token    = meta.get("eos_token", "")
chat_template = meta.get("chat_template", "")
if isinstance(chat_template, list):
    chat_template = chat_template[0].get("template", "") if chat_template else ""
pipeline_tag = meta.get("pipeline_tag", "")
transformers = meta.get("transformers_version", "")
datasets     = meta.get("datasets", [])
tags         = meta.get("tags", [])
tok_class    = meta.get("tokenizer_class", "")
tok_model_   = meta.get("tokenizer_model", "")
temperature  = meta.get("temperature", "")
top_p        = meta.get("top_p", "")
top_k        = meta.get("top_k", "")
repeat_pen   = meta.get("repeat_penalty", "")
num_predict  = meta.get("num_predict", "")
layer_count  = meta.get("layer_count", "")
vocab_size   = meta.get("vocab_size", "")
embedding    = meta.get("embedding_length", "")
heads        = meta.get("head_count", "")
rope_base    = meta.get("rope_freq_base", "")

L = []

# ── Cabecera de comentario ────────────────────────────────────
L.append("# " + "═"*60)
L.append(f"#  Modelfile — {model_name} [{quant}]")
L.append(f"#  {quant_desc}")
L.append("# " + "─"*60)
if author:       L.append(f"#  Autor        : {author}")
if license_:     L.append(f"#  Licencia     : {license_}")
if base_model:   L.append(f"#  Base model   : {base_model}")
if version:      L.append(f"#  Versión      : {version}")
if arch:         L.append(f"#  Arquitectura : {arch}")
if context_len:  L.append(f"#  Contexto     : {context_len} tokens")
if layer_count:  L.append(f"#  Capas        : {layer_count}")
if embedding:    L.append(f"#  Embedding    : {embedding}")
if heads:        L.append(f"#  Heads        : {heads}")
if vocab_size:   L.append(f"#  Vocabulario  : {vocab_size} tokens")
if rope_base:    L.append(f"#  RoPE base    : {rope_base}")
if tok_class:    L.append(f"#  Tokenizer    : {tok_class}" + (f" / {tok_model_}" if tok_model_ else ""))
if languages:    L.append(f"#  Idiomas      : {', '.join(languages)}")
if pipeline_tag: L.append(f"#  Pipeline     : {pipeline_tag}")
if datasets:     L.append(f"#  Datasets     : {', '.join(str(d) for d in datasets[:4])}")
if tags:         L.append(f"#  Tags         : {', '.join(str(t) for t in tags[:8])}")
if transformers: L.append(f"#  Transformers : {transformers}")
if description:
    L.append("#")
    for line in textwrap.wrap(description, 60):
        L.append(f"#  {line}")
L.append("# " + "═"*60)
L.append("")

# ── FROM ──────────────────────────────────────────────────────
L.append(f"FROM {os.path.realpath(gguf_path)}")
L.append("")

# ── LABEL (metadatos reales solamente) ────────────────────────
L.append("# ── Metadatos ───────────────────────────────────────────")
L.append(f'LABEL name="{model_name}"')
L.append(f'LABEL quantization="{quant}"')
if arch:         L.append(f'LABEL architecture="{arch}"')
if context_len:  L.append(f'LABEL context_length="{context_len}"')
if license_:     L.append(f'LABEL license="{license_}"')
if author:       L.append(f'LABEL author="{author}"')
if version:      L.append(f'LABEL version="{version}"')
if base_model:   L.append(f'LABEL base_model="{base_model}"')
if pipeline_tag: L.append(f'LABEL pipeline_tag="{pipeline_tag}"')
if languages:    L.append(f'LABEL languages="{",".join(languages)}"')
L.append("")

# ── PARAMETER (solo lo que viene del modelo) ──────────────────
L.append("# ── Parámetros de generación ─────────────────────────────")
L.append(f"PARAMETER num_ctx {context_len}")

if temperature:
    L.append(f"PARAMETER temperature {temperature}")
if top_p:
    L.append(f"PARAMETER top_p {top_p}")
if top_k:
    L.append(f"PARAMETER top_k {top_k}")
if repeat_pen and str(repeat_pen) not in ("1", "1.0"):
    L.append(f"PARAMETER repeat_penalty {repeat_pen}")
if num_predict:
    L.append(f"PARAMETER num_predict {num_predict}")

# ── Stop tokens (de tokenizer_config + special_tokens_map) ────
if stop_tokens:
    L.append("")
    L.append("# ── Stop tokens (tokenizer del modelo) ──────────────────")
    for st in stop_tokens:
        L.append(f'PARAMETER stop "{st}"')

# ── TEMPLATE (chat_template real del tokenizer) ───────────────
if chat_template:
    L.append("")
    L.append("# ── Chat template (tokenizer_config.json) ───────────────")
    L.append(f'TEMPLATE """{chat_template}"""')

# ── SYSTEM: no inventar nada ──────────────────────────────────
L.append("")
L.append("# ── System prompt ───────────────────────────────────────")
L.append("# Descomenta para añadir un system prompt por defecto:")
L.append('# SYSTEM """Eres un asistente útil y preciso."""')

with open(mf_path, "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")

print(mf_path)
PYEOF
    echo "$mf_path"
}

# ── Conversión base F16 ───────────────────────────────────────
F16_PATH="$MODEL_OUT_DIR/${MODEL_NAME}-${BASE_TYPE^^}.gguf"

if [ "$SKIP_CONVERT" = false ]; then
    section "Convirtiendo a ${BASE_TYPE^^} base"

    CONVERT_CMD=(
        python3 "$CONVERT_SCRIPT" "$MODEL_DIR"
        --outfile "$F16_PATH"
        --outtype "$BASE_TYPE"
        --model-name "$MODEL_NAME"
    )
    # Pasar metadatos al conversor si los hay
    if [ -n "$GGUF_META_JSON" ] && [ "$GGUF_META_JSON" != "{}" ]; then
        CONVERT_CMD+=(--metadata "$GGUF_META_JSON")
    fi

    "${CONVERT_CMD[@]}" 2>&1
    ok "${BASE_TYPE^^} generado: $F16_PATH"
else
    F16_PATH=$(find "$MODEL_OUT_DIR" -name "*.gguf" 2>/dev/null | head -1 || echo "")
    [ -z "$F16_PATH" ] && { err "No hay archivo base. Omite --skip-convert."; exit 1; }
    ok "Usando base existente: $F16_PATH"
fi

# ── Generar variantes ─────────────────────────────────────────
section "Generando variantes GGUF (${#SELECTED_QUANTS[@]} cuantizaciones)"

SUCCESS_LIST=(); FAILED_LIST=(); IDX=0
TOTAL=${#SELECTED_QUANTS[@]}

for QUANT in "${SELECTED_QUANTS[@]}"; do
    IDX=$((IDX + 1))
    OUT_FILE="$MODEL_OUT_DIR/${MODEL_NAME}-${QUANT}.gguf"

    echo -e "\n${BOLD}[$IDX/$TOTAL]${NC} ${CYAN}$QUANT${NC}  ${DIM}${QUANT_DESC[$QUANT]:-}${NC}"

    if [ -f "$OUT_FILE" ]; then
        warn "Ya existe, saltando."
        SUCCESS_LIST+=("$QUANT"); continue
    fi

    if is_base_quant "$QUANT"; then
        if [ "$QUANT" = "F16" ]; then
            cp "$F16_PATH" "$OUT_FILE"
            ok "$(basename "$OUT_FILE")"
        else
            QFLAG=$(echo "$QUANT" | tr '[:upper:]' '[:lower:]')
            CONVERT_CMD=(
                python3 "$CONVERT_SCRIPT" "$MODEL_DIR"
                --outfile "$OUT_FILE"
                --outtype "$QFLAG"
                --model-name "$MODEL_NAME"
            )
            [ -n "$GGUF_META_JSON" ] && [ "$GGUF_META_JSON" != "{}" ] && \
                CONVERT_CMD+=(--metadata "$GGUF_META_JSON")

            "${CONVERT_CMD[@]}" 2>&1 \
            && ok "$(basename "$OUT_FILE")" \
            || { err "Falló $QUANT"; FAILED_LIST+=("$QUANT"); continue; }
        fi
    else
        [ "$BASE_ONLY" = true ]     && { warn "Saltando (--base-only)"; continue; }
        [ ! -f "$QUANTIZE_BIN" ]    && { err "llama-quantize no disponible"; FAILED_LIST+=("$QUANT"); continue; }

        "$QUANTIZE_BIN" "$F16_PATH" "$OUT_FILE" "$QUANT" 2>&1 \
        && ok "$(basename "$OUT_FILE")" \
        || { err "Falló $QUANT"; FAILED_LIST+=("$QUANT"); continue; }
    fi

    MF=$(generate_modelfile "$OUT_FILE" "$QUANT" 2>/dev/null)
    echo -e "   ${DIM}Modelfile: $(basename "$MF")${NC}"
    SIZE=$(du -sh "$OUT_FILE" 2>/dev/null | cut -f1)
    echo -e "   ${DIM}Tamaño:    $SIZE${NC}"
    SUCCESS_LIST+=("$QUANT")
done

# ── README ────────────────────────────────────────────────────
section "Generando README"

python3 - "$MODEL_OUT_DIR" "$MODEL_NAME" "$META_JSON" \
    "${SUCCESS_LIST[@]+"${SUCCESS_LIST[@]}"}" <<'PYEOF'
import sys, json, os
from pathlib import Path
from datetime import datetime

out_dir    = sys.argv[1]
model_name = sys.argv[2]
meta_json  = sys.argv[3]
quants     = sys.argv[4:]

try:
    meta = json.loads(meta_json)
except Exception:
    meta = {}

quant_desc = {
    "F32":"Float 32 — máxima precisión", "F16":"Float 16 — referencia",
    "BF16":"BFloat 16", "Q8_0":"8-bit — casi sin pérdida",
    "Q6_K":"6-bit K-quant", "Q5_K_M":"5-bit K-quant medio ⭐",
    "Q5_K_S":"5-bit K-quant pequeño", "Q5_0":"5-bit legacy",
    "Q5_1":"5-bit legacy v1", "Q4_K_M":"4-bit K-quant medio ⭐",
    "Q4_K_S":"4-bit K-quant pequeño", "Q4_0":"4-bit legacy",
    "Q4_1":"4-bit legacy v1", "Q3_K_L":"3-bit K-quant grande",
    "Q3_K_M":"3-bit K-quant medio", "Q3_K_S":"3-bit K-quant pequeño",
    "Q2_K":"2-bit K-quant máxima compresión",
    "IQ4_XS":"4-bit iQuant XS", "IQ4_NL":"4-bit iQuant NL",
    "IQ3_M":"3-bit iQuant M", "IQ3_S":"3-bit iQuant S",
    "IQ3_XS":"3-bit iQuant XS", "IQ2_M":"2-bit iQuant M",
    "IQ2_S":"2-bit iQuant S", "IQ2_XS":"2-bit iQuant XS",
    "IQ1_S":"1-bit iQuant S", "IQ1_M":"1-bit iQuant M",
}

L = [f"# {model_name} — Variantes GGUF", ""]

desc = meta.get("description", "")
if desc:
    L += [desc, ""]

L += ["## Información del modelo", "", "| Campo | Valor |", "|---|---|"]
for label, key in [
    ("Arquitectura", "architecture"), ("Contexto", "context_length"),
    ("Capas", "layer_count"), ("Embedding", "embedding_length"),
    ("Attention heads", "head_count"), ("Vocabulario", "vocab_size"),
    ("Autor", "author"), ("Licencia", "license"), ("Versión", "version"),
    ("Base model", "base_model"), ("Pipeline", "pipeline_tag"),
    ("Tokenizer", "tokenizer_class"),
]:
    val = meta.get(key, "")
    if val:
        suffix = " tokens" if key in ("context_length", "vocab_size") else ""
        L.append(f"| {label} | `{val}{suffix}` |")

langs = meta.get("languages", [])
if langs: L.append(f"| Idiomas | {', '.join(langs)} |")
datasets = meta.get("datasets", [])
if datasets: L.append(f"| Datasets | {', '.join(str(d) for d in datasets[:5])} |")
tags = meta.get("tags", [])
if tags: L.append(f"| Tags | {', '.join(str(t) for t in tags[:8])} |")
L.append(f"| Generado | {datetime.now().strftime('%Y-%m-%d %H:%M')} |")
L += ["", "## Variantes generadas", "",
      "| Cuantización | Descripción | Tamaño |", "|---|---|---|"]

for q in quants:
    f = Path(out_dir) / f"{model_name}-{q}.gguf"
    if f.exists():
        size = os.popen(f"du -sh '{f}' | cut -f1").read().strip()
        L.append(f"| `{q}` | {quant_desc.get(q,'')} | {size} |")

L += ["", "## Uso con Ollama", "",
      "```bash",
      f"ollama create {model_name}-Q4_K_M -f {model_name}-Q4_K_M.Modelfile",
      f"ollama run {model_name}-Q4_K_M",
      "```", "",
      "## Uso con llama.cpp", "",
      "```bash",
      f"./llama-cli -m {model_name}-Q4_K_M.gguf -p \"Hola\" -n 200",
      "```"]

Path(out_dir, "README.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"  {out_dir}/README.md")
PYEOF

# ── Resumen ───────────────────────────────────────────────────
section "Resumen"
echo -e "${GREEN}✓ Exitosos: ${#SUCCESS_LIST[@]}${NC}"
for q in "${SUCCESS_LIST[@]}"; do echo -e "  ${GREEN}·${NC} $q"; done
if [ ${#FAILED_LIST[@]} -gt 0 ]; then
    echo -e "\n${RED}✗ Fallidos: ${#FAILED_LIST[@]}${NC}"
    for q in "${FAILED_LIST[@]}"; do echo -e "  ${RED}·${NC} $q"; done
fi
echo -e "\n📦 Espacio total:"
du -sh "$MODEL_OUT_DIR/"
echo -e "\n${BOLD}${CYAN}Salida: $MODEL_OUT_DIR${NC}\n"
