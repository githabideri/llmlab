#!/usr/bin/env python3
"""
Concurrency Benchmark — tests multi-slot performance.

Sends N concurrent requests at various context depths and measures
aggregate throughput, per-slot speed, and slot starvation.

Usage:
    LLAMA_ENDPOINT=http://192.168.0.27:8080/v1/completions \
    LLAMA_SSH_HOST=llama-cpp \
    MODEL_NAME=test \
    python3 run_concurrency_test.py
"""

import json
import subprocess
import tempfile
import time
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ENDPOINT = os.getenv("LLAMA_ENDPOINT", "http://192.168.0.27:8080/v1/completions")
SSH_HOST = os.getenv("LLAMA_SSH_HOST", "llama-cpp")
MODEL_NAME = os.getenv("MODEL_NAME", "unknown")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "results")

FILL_SENTENCE = "The quick brown fox jumps over the lazy dog. "


def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def build_prompt(target_tokens: int) -> str:
    if target_tokens <= 64:
        return "What is 2+2? Answer briefly:"
    reps = max(1, int(target_tokens / 10))
    return FILL_SENTENCE * reps + "\n\nQuestion: What is the capital of France? Answer briefly:"


def run_single(endpoint: str, target_tokens: int, slot_id: int, max_tokens: int = 128) -> dict:
    """Run a single inference request. Returns timing dict."""
    prompt = build_prompt(target_tokens)
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "cache_prompt": False
    }

    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    try:
        json.dump(payload, tmp)
        tmp.close()

        start = time.time()
        result = subprocess.run(
            ["curl", "-s", "-X", "POST", endpoint,
             "-H", "Content-Type: application/json",
             "-d", f"@{tmp.name}", "-m", "300"],
            capture_output=True, text=True, timeout=310
        )
        wall = time.time() - start

        if result.returncode != 0:
            return {"slot": slot_id, "target_ctx": target_tokens, "error": f"curl exit {result.returncode}", "wall_s": wall}

        data = json.loads(result.stdout)
        if "error" in data:
            return {"slot": slot_id, "target_ctx": target_tokens, "error": str(data["error"]), "wall_s": wall}

        usage = data.get("usage", {})
        timings = data.get("timings", {})

        pp_tokens = usage.get("prompt_tokens", 0)
        tg_tokens = usage.get("completion_tokens", 0)
        pp_ms = timings.get("prompt_ms", 0)
        tg_ms = timings.get("predicted_ms", 0)
        pp_toks = round(pp_tokens / (pp_ms / 1000), 2) if pp_ms > 0 else 0
        tg_toks = round(tg_tokens / (tg_ms / 1000), 2) if tg_ms > 0 else 0

        return {
            "slot": slot_id,
            "target_ctx": target_tokens,
            "actual_ctx": pp_tokens,
            "pp_toks": pp_toks,
            "tg_toks": tg_toks,
            "pp_tokens": pp_tokens,
            "tg_tokens": tg_tokens,
            "pp_ms": pp_ms,
            "tg_ms": tg_ms,
            "wall_s": round(wall, 2)
        }
    except Exception as e:
        return {"slot": slot_id, "target_ctx": target_tokens, "error": str(e), "wall_s": 0}
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


def get_gpu_stats(ssh_host: str) -> list:
    try:
        result = subprocess.run(
            f"ssh {ssh_host} 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits'",
            shell=True, capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    gpus.append({"index": int(parts[0]), "mem_used": int(parts[1]), "util": int(parts[2])})
            return gpus
    except:
        pass
    return []


def run_test_case(endpoint: str, ssh_host: str, concurrency: int, ctx_configs: list, label: str) -> dict:
    """
    Run one concurrency test case.
    ctx_configs: list of target token counts, one per concurrent slot.
    """
    log(f"  {label}: {concurrency} concurrent, contexts={ctx_configs}")

    pre_gpu = get_gpu_stats(ssh_host)
    results = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {}
        for i, ctx in enumerate(ctx_configs):
            f = executor.submit(run_single, endpoint, ctx, i)
            futures[f] = i

        for f in as_completed(futures):
            r = f.result()
            results.append(r)

    post_gpu = get_gpu_stats(ssh_host)

    # Sort by slot
    results.sort(key=lambda r: r.get("slot", 0))

    # Summary
    ok_results = [r for r in results if "error" not in r]
    err_results = [r for r in results if "error" in r]

    agg_tg = sum(r["tg_toks"] for r in ok_results)
    min_tg = min((r["tg_toks"] for r in ok_results), default=0)
    max_tg = max((r["tg_toks"] for r in ok_results), default=0)

    for r in ok_results:
        log(f"    slot{r['slot']}: ctx={r['actual_ctx']:,} PP={r['pp_toks']:.1f} TG={r['tg_toks']:.1f} wall={r['wall_s']}s")
    for r in err_results:
        log(f"    slot{r['slot']}: ERROR {r['error']}")

    log(f"    Aggregate TG: {agg_tg:.1f} tok/s | Range: {min_tg:.1f}–{max_tg:.1f}")

    return {
        "label": label,
        "concurrency": concurrency,
        "ctx_configs": ctx_configs,
        "results": results,
        "aggregate_tg": round(agg_tg, 2),
        "min_tg": round(min_tg, 2),
        "max_tg": round(max_tg, 2),
        "gpu_post": post_gpu,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


def main():
    test_cases = [
        # (concurrency, ctx_configs, label)
        (1, [2000], "1×2K (baseline)"),
        (2, [2000, 2000], "2×2K (symmetric)"),
        (3, [2000, 2000, 2000], "3×2K (full slots)"),
        (1, [16000], "1×16K (baseline)"),
        (2, [16000, 16000], "2×16K (symmetric)"),
        (3, [16000, 16000, 16000], "3×16K (full slots)"),
        (2, [2000, 32000], "2× asymmetric (2K+32K)"),
        (2, [8000, 64000], "2× asymmetric (8K+64K)"),
        (3, [2000, 16000, 64000], "3× mixed (2K+16K+64K)"),
        (1, [64000], "1×64K (baseline)"),
        (2, [32000, 32000], "2×32K (symmetric)"),
    ]

    print("=" * 70)
    print("Concurrency Benchmark")
    print("=" * 70)
    log(f"Model:    {MODEL_NAME}")
    log(f"Endpoint: {ENDPOINT}")
    log(f"Tests:    {len(test_cases)}")
    print("=" * 70)

    all_results = []

    for conc, ctxs, label in test_cases:
        result = run_test_case(ENDPOINT, SSH_HOST, conc, ctxs, label)
        all_results.append(result)
        time.sleep(5)  # cooldown

    # Save
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    base = f"{ts}-{MODEL_NAME}-concurrency"

    json_path = Path(OUTPUT_DIR) / f"{base}.json"
    with open(json_path, 'w') as f:
        json.dump({"model": MODEL_NAME, "results": all_results}, f, indent=2)
    log(f"✓ JSON: {json_path}")

    # Markdown summary
    md_path = Path(OUTPUT_DIR) / f"{base}.md"
    with open(md_path, 'w') as f:
        f.write(f"# Concurrency Benchmark — {MODEL_NAME}\n\n")
        f.write(f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write("| Test | Slots | Agg TG | Min TG | Max TG | Notes |\n")
        f.write("|------|:-----:|-------:|-------:|-------:|-------|\n")
        for r in all_results:
            errs = sum(1 for x in r["results"] if "error" in x)
            note = f"{errs} errors" if errs else "OK"
            f.write(f"| {r['label']} | {r['concurrency']} | {r['aggregate_tg']:.1f} | "
                    f"{r['min_tg']:.1f} | {r['max_tg']:.1f} | {note} |\n")
    log(f"✓ Markdown: {md_path}")

    print("=" * 70)
    log("Done!")


if __name__ == "__main__":
    main()
