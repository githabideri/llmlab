#!/usr/bin/env python3
"""Fetch model architecture info and GGUF quant candidates from Hugging Face.

Usage:
    python3 fetch_model_info.py <model_id_or_url>

Examples:
    python3 fetch_model_info.py moonshotai/Kimi-Linear-48B-A3B-Instruct
    python3 fetch_model_info.py nvidia/Nemotron-3-Nano-30B-A3B

Environment variables (optionally via .env in this script directory):
    HF_TOKEN          - HuggingFace token (needed for gated models)
    KV_CACHE_K_BYTES  - Bytes/elem for K cache (default: 1.0 = q8_0)
    KV_CACHE_V_BYTES  - Bytes/elem for V cache (default: 0.5 = q4_0)
    MIN_CONTEXT       - Minimum viable context tokens (default: 100000)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path


# -----------------------------
# Env loading
# -----------------------------

def load_env_file() -> None:
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.split("#", 1)[0].strip()  # allow inline comments
        os.environ.setdefault(k.strip(), v)


load_env_file()


# -----------------------------
# HF API helpers
# -----------------------------

def hf_get_json(url: str, timeout: int = 20) -> dict | list:
    req = urllib.request.Request(url)
    token = os.environ.get("HF_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def parse_model_id(raw: str) -> str:
    raw = raw.strip()
    m = re.match(r"https?://huggingface\.co/([^/]+/[^/]+)/?", raw)
    if m:
        return m.group(1)
    return raw


def fetch_config(model_id: str) -> dict | None:
    """Fetch model config from HF API, fallback to raw config.json URL."""
    encoded = urllib.parse.quote(model_id, safe="/")

    api_cfg = None

    # API endpoint with expanded config
    try:
        api_url = f"https://huggingface.co/api/models/{encoded}?expand=config"
        data = hf_get_json(api_url)
        cfg = data.get("config") if isinstance(data, dict) else None
        if isinstance(cfg, dict) and cfg:
            api_cfg = cfg
    except Exception:
        pass

    # Raw config.json often contains fuller architectural fields
    try:
        raw_url = f"https://huggingface.co/{model_id}/raw/main/config.json"
        raw_cfg = hf_get_json(raw_url)
        if isinstance(raw_cfg, dict) and raw_cfg:
            return raw_cfg
    except Exception:
        pass

    # Fall back to API config if raw config unavailable
    if api_cfg:
        return api_cfg

    print(f"⚠️  Could not fetch config for {model_id}", file=sys.stderr)
    return None


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _base_candidates(model_id: str) -> list[str]:
    name = model_id.split("/")[-1]
    cands = [name]
    for suf in ("-Instruct", "-Base", "-Chat", "-Preview", "-thinking", "-Thinking"):
        if name.endswith(suf):
            cands.append(name[: -len(suf)])
    # Also trim common version tail
    cands.append(re.sub(r"-(\d{4}|v\d+)$", "", name, flags=re.IGNORECASE))
    out = []
    for c in cands:
        c = c.strip("-_")
        if c and c not in out:
            out.append(c)
    return out


def search_gguf_repos(model_id: str, providers: list[str] | None = None, limit: int = 12) -> list[dict]:
    """Search HF model index for likely GGUF quant repos for this model."""
    if providers is None:
        providers = ["mradermacher", "bartowski"]

    base_names = _base_candidates(model_id)
    base_norms = [_norm(x) for x in base_names if x]

    seen: dict[str, dict] = {}

    queries = []
    for b in base_names:
        queries.append(f"{b} GGUF")
        queries.append(b)

    for query in queries:
        q = urllib.parse.quote(query)
        url = f"https://huggingface.co/api/models?search={q}&limit=100"
        try:
            items = hf_get_json(url)
        except Exception:
            continue
        if not isinstance(items, list):
            continue

        for item in items:
            if not isinstance(item, dict):
                continue
            repo_id = item.get("id", "")
            if not repo_id:
                continue
            repo_l = repo_id.lower()
            tags = item.get("tags", []) or []
            is_gguf = ("gguf" in repo_l) or any("gguf" in str(t).lower() for t in tags)
            if not is_gguf:
                continue

            provider = repo_id.split("/", 1)[0] if "/" in repo_id else "unknown"
            rec = seen.get(repo_id)
            if rec is None:
                seen[repo_id] = {
                    "repo_id": repo_id,
                    "provider": provider,
                    "downloads": item.get("downloads", 0) or 0,
                    "likes": item.get("likes", 0) or 0,
                }

    def score(rec: dict) -> tuple:
        rid = rec["repo_id"].lower()
        provider = rec["provider"].lower()

        provider_rank = len(providers) + 1
        for i, p in enumerate(providers):
            if provider == p.lower():
                provider_rank = i
                break

        name_hit = any(b in _norm(rid) for b in base_norms)
        return (
            0 if name_hit else 1,
            provider_rank,
            -(rec.get("downloads", 0)),
            -(rec.get("likes", 0)),
            rid,
        )

    ranked = sorted(seen.values(), key=score)
    return ranked[:limit]


def list_gguf_files(repo_id: str) -> list[dict]:
    """List .gguf files in a repo with sizes via HF tree API."""
    encoded = urllib.parse.quote(repo_id, safe="/")
    url = f"https://huggingface.co/api/models/{encoded}/tree/main?recursive=1"

    try:
        tree = hf_get_json(url)
    except Exception:
        return []

    if not isinstance(tree, list):
        return []

    files: list[dict] = []
    for item in tree:
        if not isinstance(item, dict):
            continue
        path = item.get("path") or item.get("rfilename") or ""
        if not str(path).endswith(".gguf"):
            continue

        size = item.get("size")
        if not isinstance(size, int):
            size = item.get("lfs", {}).get("size", 0)
        if not isinstance(size, int):
            size = 0

        files.append(
            {
                "filename": str(path),
                "size_bytes": size,
                "size_human": human_size(size),
            }
        )

    return sorted(files, key=lambda x: (x["size_bytes"], x["filename"]))


# -----------------------------
# Calculations
# -----------------------------

def human_size(nbytes: int) -> str:
    val = float(nbytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if val < 1024:
            return f"{val:.1f} {unit}"
        val /= 1024
    return f"{val:.1f} PB"


def calc_kv_per_token(config: dict) -> dict:
    n_layers = int(config.get("num_hidden_layers", 0) or 0)
    n_heads = int(config.get("num_attention_heads", 0) or 0)
    n_kv_heads = int(config.get("num_key_value_heads", n_heads) or 0)
    hidden_size = int(config.get("hidden_size", 0) or 0)
    head_dim = int(config.get("head_dim", hidden_size // n_heads if n_heads else 0) or 0)

    n_attn_layers = n_layers  # default assumption
    notes: list[str] = []

    arch = str(config.get("model_type", "")).lower()
    cfg_str = json.dumps(config).lower()
    if "mamba" in arch or "ssm" in cfg_str:
        notes.append("⚠️ Interleaved Mamba/attention architecture detected: verify attention-layer count manually.")
    if "mla" in cfg_str or "deepseek" in arch:
        notes.append("⚠️ Possible MLA/compressed KV: standard KV formula may overestimate cache.")

    k_bytes = float(os.environ.get("KV_CACHE_K_BYTES", "1.0"))
    v_bytes = float(os.environ.get("KV_CACHE_V_BYTES", "0.5"))

    kv_bytes = n_attn_layers * n_kv_heads * head_dim * (k_bytes + v_bytes)
    return {
        "n_layers": n_layers,
        "n_attn_layers": n_attn_layers,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "hidden_size": hidden_size,
        "head_dim": head_dim,
        "k_bytes_per_elem": k_bytes,
        "v_bytes_per_elem": v_bytes,
        "kv_bytes_per_token": kv_bytes,
        "kv_kib_per_token": kv_bytes / 1024 if kv_bytes > 0 else 0,
        "notes": notes,
    }


def calc_fitment(model_size_gib: float, kv_kib_per_token: float, vram_pools_gb: list[float]) -> list[dict]:
    min_ctx = int(os.environ.get("MIN_CONTEXT", "100000"))
    out = []
    for vram in vram_pools_gb:
        avail = vram - model_size_gib
        if avail <= 0 or kv_kib_per_token <= 0:
            out.append({"vram_gb": vram, "available_gb": 0, "max_context": 0, "viable": False})
            continue
        max_ctx = int((avail * 1024 * 1024) / kv_kib_per_token)
        out.append(
            {
                "vram_gb": vram,
                "available_gb": round(avail, 2),
                "max_context": max_ctx,
                "viable": max_ctx >= min_ctx,
            }
        )
    return out


# Quant quality rank (higher = better quality)
QUANT_QUALITY_ORDER: list[tuple[str, int]] = [
    ("Q8_0", 100),
    ("Q6_K", 90),
    ("Q5_K_M", 84),
    ("Q5_K_L", 83),
    ("Q5_K_S", 82),
    ("Q4_K_M", 76),
    ("Q4_K_L", 75),
    ("IQ4_NL", 74),
    ("Q4_K_S", 73),
    ("IQ4_XS", 72),
    ("Q4_1", 70),
    ("Q4_0", 69),
    ("Q3_K_L", 62),
    ("Q3_K_M", 61),
    ("IQ3_M", 60),
    ("Q3_K_S", 59),
    ("IQ3_XS", 58),
    ("IQ3_XXS", 57),
    ("Q2_K", 50),
    ("Q2_K_S", 49),
    ("IQ2_M", 48),
    ("IQ2_S", 47),
    ("IQ2_XS", 46),
    ("IQ2_XXS", 45),
    ("IQ1_M", 40),
    ("IQ1_S", 39),
]


def quant_tier(filename: str) -> tuple[str, int, bool]:
    name = Path(filename).name.upper()
    i1 = ".I1-" in name or "I1-" in name

    for label, score in QUANT_QUALITY_ORDER:
        if label in name:
            # tiny tie-break bonus for i1 variant within same quant class
            return label, score + (1 if i1 else 0), i1
    return "UNKNOWN", 0, i1


def recommend_quants(files: list[dict], kv: dict | None) -> dict | None:
    if not kv or kv.get("kv_kib_per_token", 0) <= 0:
        return None

    viable: list[dict] = []
    all_parsed: list[dict] = []  # Track ALL parsed files for comparison
    
    for f in files:
        if f.get("size_bytes", 0) <= 0:
            continue

        name_u = Path(f["filename"]).name.upper()
        if "IMATRIX" in name_u:
            continue  # helper matrix file, not runnable quant

        size_gib = f["size_bytes"] / (1024 ** 3)
        fit = calc_fitment(size_gib, kv["kv_kib_per_token"], [23.5, 35.5])
        
        qlabel, qscore, i1 = quant_tier(f["filename"])
        if qlabel == "UNKNOWN":
            continue

        parsed = {
            **f,
            "quant_label": qlabel,
            "quant_score": qscore,
            "is_i1": i1,
            "size_gib": size_gib,
            "ctx24": fit[0]["max_context"] if fit else 0,
            "ctx36": fit[1]["max_context"] if fit else 0,
            "viable_24gb": fit[0]["viable"] if fit else False,
        }
        all_parsed.append(parsed)
        
        if fit and fit[0]["viable"]:
            viable.append(parsed)

    if not viable:
        return None

    # Quality-first: highest quant tier that still passes 100K context at 24GB
    quality = sorted(viable, key=lambda x: (x["quant_score"], x["size_bytes"]), reverse=True)[0]

    # Balanced: prefer practical Q4-ish defaults when available
    balanced_pref = ["Q4_K_M", "IQ4_NL", "Q4_K_L", "Q4_K_S", "IQ4_XS", "Q5_K_M", "Q5_K_S"]
    balanced = None
    for p in balanced_pref:
        found = next((x for x in viable if x["quant_label"] == p), None)
        if found:
            balanced = found
            break
    if balanced is None:
        balanced = quality

    # Speed-first: smallest viable quant
    speed = sorted(viable, key=lambda x: x["size_bytes"])[0]

    return {
        "quality": quality,
        "balanced": balanced,
        "speed": speed,
        "viable_count": len(viable),
        "all_candidates": all_parsed,  # For comparison table
    }


# -----------------------------
# Report formatting
# -----------------------------

def format_report(model_id: str, config: dict | None, kv: dict | None,
                  repos: list[dict], quant_files: dict[str, list[dict]]) -> str:
    lines: list[str] = [
        f"# Model Info: {model_id}",
        f"- **Model link:** https://huggingface.co/{model_id}",
        "",
    ]

    if config:
        lines += [
            "## Architecture",
            f"- **Type:** {config.get('model_type', 'unknown')}",
            f"- **Training context:** {config.get('max_position_embeddings', '?')}",
            f"- **Layers:** {config.get('num_hidden_layers', '?')}",
            f"- **Hidden size:** {config.get('hidden_size', '?')}",
            f"- **Attention heads:** {config.get('num_attention_heads', '?')}",
            f"- **KV heads:** {config.get('num_key_value_heads', config.get('num_attention_heads', '?'))}",
            f"- **Head dim:** {config.get('head_dim', '?')}",
        ]
        n_exp = config.get("num_experts") or config.get("num_local_experts")
        if n_exp:
            active = config.get("num_experts_per_tok") or config.get("num_selected_experts") or "?"
            lines.append(f"- **MoE:** {n_exp} experts, {active} active/token")
        lines.append("")

    if kv:
        lines += [
            "## KV Cache",
            f"- **Formula:** attn_layers × kv_heads × head_dim × ({kv['k_bytes_per_elem']} + {kv['v_bytes_per_elem']})",
            f"- **KV per token:** {kv['kv_bytes_per_token']:.0f} bytes ({kv['kv_kib_per_token']:.2f} KiB)",
        ]
        for n in kv.get("notes", []):
            lines.append(f"- {n}")
        lines.append("")

    if repos:
        lines.append("## GGUF Repos (ranked)")
        for r in repos:
            lines.append(f"- [{r['repo_id']}](https://huggingface.co/{r['repo_id']}) (provider: {r['provider']}, downloads: {r.get('downloads', 0)})")
        lines.append("")
    else:
        lines += [
            "## GGUF Repos (ranked)",
            "- None auto-discovered. Use manual HF search with `<model name> GGUF`.",
            "",
        ]

    if quant_files:
        lines += [
            "## Quant Files & Fitment",
            "",
        ]
        for repo_id, files in quant_files.items():
            lines.append(f"### {repo_id}")
            if not files:
                lines.append("(no .gguf files found)")
                lines.append("")
                continue

            lines.append("| File | Size | Max ctx @23.5GB | Max ctx @35.5GB | Viable @24GB? |")
            lines.append("|---|---:|---:|---:|:---:|")

            for f in files:
                if kv and kv["kv_kib_per_token"] > 0 and f["size_bytes"] > 0:
                    size_gib = f["size_bytes"] / (1024 ** 3)
                    fit = calc_fitment(size_gib, kv["kv_kib_per_token"], [23.5, 35.5])
                    c24 = f"{fit[0]['max_context']:,}" if fit[0]["max_context"] else "❌"
                    c36 = f"{fit[1]['max_context']:,}" if fit[1]["max_context"] else "❌"
                    viable = "✅" if fit[0]["viable"] else "❌"
                else:
                    c24 = c36 = viable = "?"
                lines.append(f"| `{f['filename']}` | {f['size_human']} | {c24} | {c36} | {viable} |")

            rec = recommend_quants(files, kv)
            if rec:
                lines.append("")
                lines.append("**Recommendation Candidates (sorted by quality):**")
                lines.append("")
                
                # Show comparison table for top candidates
                lines.append("| Quant | File Size | Fits @24GB? | Max Ctx @24GB | Reason |")
                lines.append("|---|---:|:---:|---:|---|")
                
                candidates = sorted(rec["all_candidates"], key=lambda x: x["quant_score"], reverse=True)
                q_best = rec["quality"]
                
                for c in candidates[:5]:  # Show top 5
                    if c["viable_24gb"]:
                        fits = "✅"
                        ctx_text = f"{c['ctx24']:,}"
                        if c["quant_label"] == q_best["quant_label"]:
                            reason = "**SELECTED** — highest quality that fits"
                        else:
                            reason = "Passes, but lower quality than best"
                    else:
                        fits = "❌"
                        # Show why it fails
                        if c["ctx24"] > 0 and c["ctx24"] < 100000:
                            ctx_text = f"{c['ctx24']:,}"
                            reason = f"Only reaches {c['ctx24']:,} ctx (need 100K)"
                        else:
                            ctx_text = "–"
                            reason = f"File {c['size_human']} > 24 GB VRAM"
                    
                    i1_suffix = " (i1)" if c["is_i1"] else ""
                    lines.append(
                        f"| {c['quant_label']}{i1_suffix} | {c['size_human']} | {fits} | {ctx_text} | {reason} |"
                    )
                
                lines.append("")
                q = rec["quality"]
                lines.append(f"**✅ Recommended:** `{q['quant_label']}` ({q['size_human']}) — {q['ctx24']:,} ctx @ 24GB")
                lines.append(f"   - File size: {q['size_human']} ✅ fits in 24GB")
                lines.append(f"   - Max context: {q['ctx24']:,} tokens ✅ exceeds 100K minimum")
                lines.append(f"   - Quality: {q['quant_label']} is highest tier passing both filters")
                
                # Show alternatives
                alts = [c for c in rec["all_candidates"] 
                        if c["viable_24gb"] and c["quant_label"] != q["quant_label"]]
                if alts:
                    lines.append("")
                    lines.append("**Alternative options:**")
                    for alt in alts[:2]:
                        lines.append(f"   - `{alt['quant_label']}` ({alt['size_human']}): {alt['ctx24']:,} ctx @ 24GB")
                
            else:
                lines.append("")
                lines.append("- Recommendation unavailable: missing KV/token data or no quant passes 100K context at 24GB.")

            lines.append("")

    return "\n".join(lines)


# -----------------------------
# CLI entrypoint
# -----------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="Fetch model info + GGUF quant fitment from HF")
    p.add_argument("model", help="HF model id or URL")
    p.add_argument("--quant-provider", "-p", action="append", default=None,
                   help="Preferred quant provider(s), repeatable (default: mradermacher, bartowski)")
    p.add_argument("--repo-limit", type=int, default=6, help="How many GGUF repos to analyze (default 6)")
    p.add_argument("--json", action="store_true", help="Output JSON")
    args = p.parse_args()

    model_id = parse_model_id(args.model)
    print(f"🔍 Fetching info for: {model_id}", file=sys.stderr)

    config = fetch_config(model_id)
    kv = calc_kv_per_token(config) if config else None

    print("🔍 Searching GGUF repos...", file=sys.stderr)
    repos = search_gguf_repos(model_id, providers=args.quant_provider, limit=max(1, args.repo_limit))

    quant_files: dict[str, list[dict]] = {}
    quant_recommendations: dict[str, dict | None] = {}
    for repo in repos:
        rid = repo["repo_id"]
        print(f"📦 Listing .gguf files in {rid}...", file=sys.stderr)
        files = list_gguf_files(rid)
        quant_files[rid] = files
        quant_recommendations[rid] = recommend_quants(files, kv)

    if args.json:
        print(json.dumps({
            "model_id": model_id,
            "model_link": f"https://huggingface.co/{model_id}",
            "config": config,
            "kv_cache": kv,
            "gguf_repos": repos,
            "quant_files": quant_files,
            "quant_recommendations": quant_recommendations,
        }, indent=2))
    else:
        print(format_report(model_id, config, kv, repos, quant_files))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
