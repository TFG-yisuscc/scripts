#!/usr/bin/env python3
"""
extract_metadata.py
Lee TODOS los metadatos disponibles de un directorio de modelo HuggingFace:
  config.json, tokenizer_config.json, generation_config.json,
  special_tokens_map.json, tokenizer.json, README.md / README
y los emite como JSON para que el script bash los consuma.

Uso:
  python3 extract_metadata.py <directorio_modelo> [--pretty]
"""

import json
import sys
import os
import re
from pathlib import Path

# ── PyYAML opcional (para parsear frontmatter del README) ─────
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# ── safetensors opcional (para metadatos del header) ──────────
try:
    import struct

    def read_safetensors_metadata(path):
        """Lee el header JSON de un archivo .safetensors"""
        with open(path, "rb") as f:
            length_bytes = f.read(8)
            if len(length_bytes) < 8:
                return {}
            header_len = struct.unpack("<Q", length_bytes)[0]
            header_bytes = f.read(header_len)
            header = json.loads(header_bytes.decode("utf-8"))
            return header.get("__metadata__", {})
except Exception:
    def read_safetensors_metadata(path):
        return {}


def load_json(path):
    """Carga JSON si existe, retorna dict vacío si no."""
    p = Path(path)
    if p.exists():
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def parse_readme_frontmatter(model_dir):
    """
    Extrae el YAML frontmatter del README.md / README.
    Retorna dict con los campos encontrados.
    """
    meta = {}
    for name in ["README.md", "README", "readme.md"]:
        readme = Path(model_dir) / name
        if not readme.exists():
            continue

        text = readme.read_text(encoding="utf-8", errors="replace")

        # Extraer bloque YAML entre ---
        fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
        if fm_match:
            fm_text = fm_match.group(1)
            if HAS_YAML:
                try:
                    meta = yaml.safe_load(fm_text) or {}
                except Exception:
                    meta = _parse_simple_yaml(fm_text)
            else:
                meta = _parse_simple_yaml(fm_text)

        # Extraer descripción del cuerpo (primer párrafo no vacío tras el frontmatter)
        body = text[fm_match.end():] if fm_match else text
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
        # Saltar headings
        for p in paragraphs:
            if not p.startswith("#") and len(p) > 20:
                meta["_readme_description"] = p.replace("\n", " ")[:500]
                break

        break  # Solo el primer README encontrado
    return meta


def _parse_simple_yaml(text):
    """Parser YAML mínimo para cuando PyYAML no está disponible."""
    result = {}
    for line in text.splitlines():
        if ":" in line and not line.strip().startswith("-"):
            k, _, v = line.partition(":")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if v:
                result[k] = v
    return result


def normalize_token(tok):
    """Extrae el string de un token que puede ser str o dict."""
    if isinstance(tok, str):
        return tok
    if isinstance(tok, dict):
        return tok.get("content", tok.get("id", ""))
    return ""


def collect_stop_tokens(cfg, tok_cfg, gen_cfg, sp_tokens):
    """Recopila todos los stop tokens de múltiples fuentes."""
    stops = set()

    # Desde generation_config
    eos_ids = gen_cfg.get("eos_token_id", [])
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]

    # Desde tokenizer_config
    for key in ["eos_token", "pad_token"]:
        tok = normalize_token(tok_cfg.get(key, ""))
        if tok:
            stops.add(tok)

    # Desde special_tokens_map
    for key in ["eos_token", "pad_token", "sep_token"]:
        tok = normalize_token(sp_tokens.get(key, ""))
        if tok:
            stops.add(tok)

    # Chat template — extraer tokens de fin de turno comunes
    template = tok_cfg.get("chat_template", "")
    if isinstance(template, list):
        # Algunos modelos tienen lista de templates
        for t in template:
            if isinstance(t, dict):
                template = t.get("template", "")
                break

    if template:
        # Buscar tokens literales entre comillas en el template
        candidates = re.findall(r"'(<[^']+>)'|\"(<[^\"]+>)\"", str(template))
        for groups in candidates:
            for g in groups:
                if g and g not in {"<s>", "</s>", "<unk>"}:
                    stops.add(g)

        # Tokens especiales de fin de turno comunes
        end_markers = [
            "<|eot_id|>", "<|end_of_text|>", "<|im_end|>", "<end_of_turn>",
            "<|endoftext|>", "<|end_of_sentence|>", "<|EOT|>",
            "<|end_of_role|>", "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>",
            "<|user|>", "<|assistant|>", "<|system|>",
        ]
        for marker in end_markers:
            if marker in str(template):
                stops.add(marker)

    return sorted(stops)


def extract_all_metadata(model_dir):
    """
    Extrae metadatos de todas las fuentes disponibles.
    Devuelve un dict estructurado.
    """
    model_dir = Path(model_dir)
    meta = {}

    # ── 1. config.json ───────────────────────────────────────
    cfg = load_json(model_dir / "config.json")
    meta["_sources"] = {"config": bool(cfg)}

    # Arquitectura
    arch = cfg.get("model_type") or cfg.get("architectures", [None])[0]
    if arch:
        meta["architecture"] = arch

    # Parámetros de arquitectura
    for key, dest in [
        ("max_position_embeddings", "context_length"),
        ("context_length", "context_length"),
        ("max_sequence_length", "context_length"),
        ("hidden_size", "embedding_length"),
        ("intermediate_size", "feed_forward_length"),
        ("num_attention_heads", "head_count"),
        ("num_key_value_heads", "head_count_kv"),
        ("num_hidden_layers", "layer_count"),
        ("rope_theta", "rope_freq_base"),
        ("rms_norm_eps", "layer_norm_rms_epsilon"),
        ("vocab_size", "vocab_size"),
        ("rope_scaling", "rope_scaling"),
        ("sliding_window", "sliding_window"),
        ("head_dim", "head_dim"),
        ("num_experts", "expert_count"),
        ("num_experts_per_tok", "expert_used_count"),
    ]:
        val = cfg.get(key)
        if val is not None:
            meta[dest] = val

    # torch_dtype → tipo base
    if "torch_dtype" in cfg:
        meta["file_type_hint"] = cfg["torch_dtype"]

    # Metadatos del modelo en config.json
    for key in ["_name_or_path", "name_or_path"]:
        if key in cfg:
            meta["source_path"] = cfg[key]

    meta["_sources"]["config"] = bool(cfg)

    # ── 2. tokenizer_config.json ─────────────────────────────
    tok_cfg = load_json(model_dir / "tokenizer_config.json")
    meta["_sources"]["tokenizer_config"] = bool(tok_cfg)

    if tok_cfg:
        # Chat template
        chat_template = tok_cfg.get("chat_template")
        if chat_template:
            if isinstance(chat_template, list):
                # Elegir el template "default" o el primero
                for t in chat_template:
                    if isinstance(t, dict) and t.get("name") in ("default", "chat", None):
                        chat_template = t.get("template", "")
                        break
                if isinstance(chat_template, list):
                    chat_template = chat_template[0].get("template", "") if chat_template else ""
            meta["chat_template"] = chat_template

        # Tokens especiales
        for key in ["bos_token", "eos_token", "pad_token", "unk_token",
                    "sep_token", "mask_token"]:
            tok = normalize_token(tok_cfg.get(key, ""))
            if tok:
                meta[key] = tok

        # Clase de tokenizer
        if "tokenizer_class" in tok_cfg:
            meta["tokenizer_class"] = tok_cfg["tokenizer_class"]

        # model_max_length como contexto de respaldo
        if "model_max_length" in tok_cfg and "context_length" not in meta:
            val = tok_cfg["model_max_length"]
            if isinstance(val, (int, float)) and val < 10_000_000:
                meta["context_length"] = int(val)

        # add_bos_token / add_eos_token
        for key in ["add_bos_token", "add_eos_token", "add_prefix_space"]:
            if key in tok_cfg:
                meta[key] = tok_cfg[key]

    # ── 3. generation_config.json ────────────────────────────
    gen_cfg = load_json(model_dir / "generation_config.json")
    meta["_sources"]["generation_config"] = bool(gen_cfg)

    if gen_cfg:
        # Parámetros de generación (solo si son valores no triviales)
        for key, dest, default in [
            ("temperature",        "temperature",     None),
            ("top_p",             "top_p",            None),
            ("top_k",             "top_k",            None),
            ("repetition_penalty","repeat_penalty",   1.0),
            ("max_new_tokens",    "num_predict",      None),
            ("max_length",        "num_predict",      None),
            ("do_sample",         "do_sample",        None),
        ]:
            val = gen_cfg.get(key)
            if val is not None and val != default:
                meta[dest] = val

        # Transformers version
        if "_transformers_version" in gen_cfg:
            meta["transformers_version"] = gen_cfg["_transformers_version"]

    # ── 4. special_tokens_map.json ───────────────────────────
    sp_tokens = load_json(model_dir / "special_tokens_map.json")
    meta["_sources"]["special_tokens_map"] = bool(sp_tokens)

    # Rellenar tokens que no vinieron de tokenizer_config
    for key in ["bos_token", "eos_token", "pad_token", "unk_token"]:
        if key not in meta:
            tok = normalize_token(sp_tokens.get(key, ""))
            if tok:
                meta[key] = tok

    # ── 5. tokenizer.json ────────────────────────────────────
    tok_json = load_json(model_dir / "tokenizer.json")
    meta["_sources"]["tokenizer_json"] = bool(tok_json)

    if tok_json:
        # Vocab size real
        if "model" in tok_json:
            tok_model = tok_json["model"]
            if "vocab" in tok_model and "vocab_size" not in meta:
                meta["vocab_size"] = len(tok_model["vocab"])
            if "type" in tok_model:
                meta["tokenizer_model"] = tok_model["type"]  # BPE, Unigram...

        # Added tokens especiales
        added = tok_json.get("added_tokens", [])
        special_added = [t["content"] for t in added
                         if t.get("special") and t.get("content")]
        if special_added:
            meta["added_special_tokens"] = special_added

    # ── 6. Metadatos del header safetensors ──────────────────
    sf_meta = {}
    for sf_file in sorted(model_dir.glob("*.safetensors"))[:1]:
        sf_meta = read_safetensors_metadata(sf_file)
        break
    meta["_sources"]["safetensors_header"] = bool(sf_meta)
    if sf_meta:
        meta["safetensors_meta"] = sf_meta

    # ── 7. README.md / model card ────────────────────────────
    readme_meta = parse_readme_frontmatter(model_dir)
    meta["_sources"]["readme"] = bool(readme_meta)

    if readme_meta:
        # Licencia
        for key in ["license", "licence"]:
            if key in readme_meta and "license" not in meta:
                meta["license"] = str(readme_meta[key])

        # Lenguajes
        if "language" in readme_meta:
            langs = readme_meta["language"]
            if isinstance(langs, list):
                meta["languages"] = langs
            elif langs:
                meta["languages"] = [str(langs)]

        # Tags
        if "tags" in readme_meta:
            meta["tags"] = readme_meta["tags"] if isinstance(readme_meta["tags"], list) else [readme_meta["tags"]]

        # Nombre / base_model
        for key in ["base_model", "base model"]:
            if key in readme_meta and "base_model" not in meta:
                meta["base_model"] = str(readme_meta[key])

        # Pipeline tag
        if "pipeline_tag" in readme_meta:
            meta["pipeline_tag"] = readme_meta["pipeline_tag"]

        # Dataset info
        if "datasets" in readme_meta:
            meta["datasets"] = readme_meta["datasets"]

        # Descripción del README
        if "_readme_description" in readme_meta:
            meta["description"] = readme_meta["_readme_description"]

    # ── 8. Stop tokens (multi-fuente) ────────────────────────
    stop_tokens = collect_stop_tokens(cfg, tok_cfg, gen_cfg, sp_tokens)
    if stop_tokens:
        meta["stop_tokens"] = stop_tokens

    # ── 9. Nombre del modelo (inferido) ──────────────────────
    # Intentar obtener un nombre limpio desde múltiples fuentes
    name_candidates = [
        readme_meta.get("model_name"),
        readme_meta.get("name"),
        cfg.get("_name_or_path", "").split("/")[-1] if cfg.get("_name_or_path") else None,
        sf_meta.get("model_name"),
        sf_meta.get("name"),
    ]
    for name in name_candidates:
        if name and name.strip():
            meta["model_name_from_files"] = name.strip()
            break

    # ── 10. Autor (inferido) ─────────────────────────────────
    for key in ["author", "created_by", "organization"]:
        if key in readme_meta:
            meta["author"] = str(readme_meta[key])
            break

    # ── 11. Versión ──────────────────────────────────────────
    for key in ["version", "model_version"]:
        if key in readme_meta:
            meta["version"] = str(readme_meta[key])
            break

    # ── Limpiar internos ─────────────────────────────────────
    # Quitar claves que empiezan con _ (internas) excepto _sources
    meta_clean = {k: v for k, v in meta.items() if not k.startswith("_") or k == "_sources"}

    # Resumen de fuentes para debug
    meta_clean["_sources"] = meta["_sources"]

    return meta_clean


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 extract_metadata.py <directorio_modelo> [--pretty]", file=sys.stderr)
        sys.exit(1)

    model_dir = sys.argv[1]
    pretty = "--pretty" in sys.argv

    if not os.path.isdir(model_dir):
        print(f"Error: {model_dir} no es un directorio", file=sys.stderr)
        sys.exit(1)

    result = extract_all_metadata(model_dir)

    indent = 2 if pretty else None
    print(json.dumps(result, indent=indent, ensure_ascii=False, default=str))
