#!/usr/bin/env python3
"""
Context Ladder Benchmark Script

Tests model performance across various context lengths.
Captures PP/TG tok/s, VRAM usage, CPU/RAM usage, and timing metrics.

Uses temp files for large payloads to avoid OOM from stdin pipes.

Usage:
    ./run_context_ladder.py [--config path/to/.env]
    
Environment variables:
    LLAMA_ENDPOINT  - completions endpoint (default: http://192.168.0.27:8080/v1/completions)
    LLAMA_SSH_HOST  - SSH host for GPU stats (default: llama-cpp)
    MODEL_NAME      - model identifier for filenames
    OUTPUT_DIR      - results directory (default: results)
    TEST_POINTS     - comma-separated token counts (optional override)
    MAX_CTX         - max slot context to respect (default: 131072)
"""

import json
import subprocess
import tempfile
import time
import sys
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional

# Load .env if python-dotenv available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configuration from environment
ENDPOINT = os.getenv("LLAMA_ENDPOINT", "http://192.168.0.27:8080/v1/completions")
SSH_HOST = os.getenv("LLAMA_SSH_HOST", "llama-cpp")
MODEL_NAME = os.getenv("MODEL_NAME", "unknown-model")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "results")
MAX_CTX = int(os.getenv("MAX_CTX", "131072"))

# Default test points (in tokens) — auto-filtered by MAX_CTX
DEFAULT_TEST_POINTS = [
    64, 2000, 4000, 8000, 16000, 32000, 64000, 96000, 128000
]

# The fill sentence is ~10 tokens
FILL_SENTENCE = "The quick brown fox jumps over the lazy dog. "
FILL_TOKENS_PER_REP = 10  # approximate


def log(msg: str):
    """Timestamped log line."""
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def get_process_stats(ssh_host: str, process_name: str = "llama-server") -> Dict:
    """Get CPU and RAM usage of llama-server process via SSH."""
    cmd = f"ssh {ssh_host} \"ps aux | grep {process_name} | grep -v grep | awk '{{print $3,$4,$5,$6}}'\""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split('\n')[0].split()
            if len(parts) >= 4:
                cpu, mem_pct, vsz, rss = parts[:4]
                return {
                    "cpu_percent": float(cpu),
                    "mem_percent": float(mem_pct),
                    "vsz_kb": int(vsz),
                    "rss_kb": int(rss),
                    "rss_mb": round(int(rss) / 1024, 1)
                }
    except Exception as e:
        log(f"  WARN: process stats failed: {e}")
    return {}


def get_gpu_stats(ssh_host: str) -> List[Dict]:
    """Get GPU memory usage via nvidia-smi over SSH."""
    cmd = f"ssh {ssh_host} \"nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits\""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 5:
                        gpus.append({
                            "index": int(parts[0]),
                            "mem_used_mb": int(parts[1]),
                            "mem_total_mb": int(parts[2]),
                            "util_percent": int(parts[3]),
                            "power_watts": float(parts[4])
                        })
            return gpus
    except Exception as e:
        log(f"  WARN: GPU stats failed: {e}")
    return []


def count_tokens(endpoint: str, text: str) -> Optional[int]:
    """Use the server's tokenize endpoint to get exact token count."""
    tok_url = endpoint.replace("/v1/completions", "/tokenize")
    try:
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump({"content": text}, tmp)
        tmp.close()
        result = subprocess.run(
            ["curl", "-s", "-X", "POST", tok_url,
             "-H", "Content-Type: application/json",
             "-d", f"@{tmp.name}", "-m", "30"],
            capture_output=True, text=True, timeout=35
        )
        os.unlink(tmp.name)
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            tokens = data.get("tokens", [])
            return len(tokens)
    except Exception as e:
        log(f"  WARN: tokenize failed: {e}")
    return None


def build_prompt(target_tokens: int, endpoint: str = None) -> str:
    """Build a prompt of approximately target_tokens length.
    
    Uses conservative estimate (11 tokens per sentence rep) and
    optionally verifies with server tokenizer. Leaves 256-token
    headroom for generation + overhead.
    """
    if target_tokens <= 64:
        return "What is 2+2? Answer briefly:"
    
    # Conservative: assume ~11 tokens per fill sentence (slightly over to be safe)
    effective_target = target_tokens - 256  # headroom
    reps = max(1, int(effective_target / 11))
    padding = FILL_SENTENCE * reps
    prompt = f"{padding}\n\nQuestion: What is the capital of France? Answer briefly:"
    
    # If endpoint available, verify token count and trim if needed
    if endpoint:
        actual = count_tokens(endpoint, prompt)
        if actual is not None:
            log(f"  Tokenized: {actual:,} tokens (target: {target_tokens:,}, headroom: {target_tokens - actual})")
            # If over target, trim down
            while actual and actual > target_tokens - 128 and reps > 1:
                reps = int(reps * 0.95)
                padding = FILL_SENTENCE * reps
                prompt = f"{padding}\n\nQuestion: What is the capital of France? Answer briefly:"
                actual = count_tokens(endpoint, prompt)
                if actual:
                    log(f"  Adjusted: {actual:,} tokens (reps={reps})")
    
    return prompt


def run_inference(endpoint: str, prompt: str, max_tokens: int = 128) -> Optional[Dict]:
    """Run inference using a temp file for the payload (safe for large prompts)."""
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "cache_prompt": False
    }
    
    # Write payload to temp file to avoid pipe OOM
    tmp = None
    try:
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(payload, tmp)
        tmp.close()
        
        payload_size_mb = os.path.getsize(tmp.name) / (1024 * 1024)
        log(f"  Payload: {payload_size_mb:.1f} MiB on disk")
        
        result = subprocess.run(
            ["curl", "-s", "-X", "POST", endpoint,
             "-H", "Content-Type: application/json",
             "-d", f"@{tmp.name}",
             "-m", "600"],
            capture_output=True, text=True, timeout=620
        )
        
        if result.returncode != 0:
            log(f"  ERROR: curl exit {result.returncode}: {result.stderr[:300]}")
            return None
        
        if not result.stdout.strip():
            log(f"  ERROR: empty response")
            return None
        
        data = json.loads(result.stdout)
        
        # Check for server error
        if "error" in data:
            log(f"  ERROR: server returned: {data['error'].get('message', data['error'])[:200]}")
            return None
        
        return data
        
    except json.JSONDecodeError as e:
        log(f"  ERROR: invalid JSON response: {e}")
        return None
    except subprocess.TimeoutExpired:
        log(f"  ERROR: curl timed out after 600s")
        return None
    except Exception as e:
        log(f"  ERROR: {e}")
        return None
    finally:
        if tmp and os.path.exists(tmp.name):
            os.unlink(tmp.name)


def test_context_point(
    endpoint: str,
    ssh_host: str,
    target_tokens: int,
) -> Optional[Dict]:
    """Test a single context point and gather all metrics."""
    
    log(f"--- Testing ~{target_tokens:,} tokens ---")
    
    # Build prompt (with token count verification)
    prompt = build_prompt(target_tokens, endpoint)
    prompt_chars = len(prompt)
    log(f"  Prompt: {prompt_chars:,} chars (~{target_tokens:,} target tokens)")
    
    # Get baseline stats
    pre_gpu = get_gpu_stats(ssh_host)
    
    # Run inference
    start_time = time.time()
    data = run_inference(endpoint, prompt)
    end_time = time.time()
    wall_time = round(end_time - start_time, 2)
    
    if not data:
        log(f"  FAILED after {wall_time}s")
        return None
    
    # Get post-inference stats
    post_gpu = get_gpu_stats(ssh_host)
    
    # Extract metrics
    usage = data.get("usage", {})
    timings = data.get("timings", {})
    
    pp_tokens = usage.get("prompt_tokens", 0)
    tg_tokens = usage.get("completion_tokens", 0)
    pp_ms = timings.get("prompt_ms", 0)
    tg_ms = timings.get("predicted_ms", 0)
    
    pp_toks = round(pp_tokens / (pp_ms / 1000), 2) if pp_ms > 0 else 0
    tg_toks = round(tg_tokens / (tg_ms / 1000), 2) if tg_ms > 0 else 0
    
    # VRAM summary
    vram_str = ""
    if post_gpu:
        vram_parts = [f"GPU{g['index']}:{g['mem_used_mb']}MiB" for g in post_gpu]
        vram_str = " | ".join(vram_parts)
    
    result = {
        "target_ctx": target_tokens,
        "actual_ctx": pp_tokens,
        "pp_toks": pp_toks,
        "tg_toks": tg_toks,
        "pp_tokens": pp_tokens,
        "tg_tokens": tg_tokens,
        "pp_ms": pp_ms,
        "tg_ms": tg_ms,
        "wall_time_s": wall_time,
        "gpu_post": post_gpu,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    log(f"  ✓ {pp_tokens:>6,} tokens | PP={pp_toks:>8.1f} tok/s | TG={tg_toks:>6.1f} tok/s | wall={wall_time}s")
    if vram_str:
        log(f"  VRAM: {vram_str}")
    
    return result


def save_results(results: List[Dict], model_name: str, output_dir: str):
    """Save results in JSON, CSV, and Markdown."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    base_name = f"{timestamp}-{model_name.replace('/', '-')}"
    
    # JSON
    json_path = Path(output_dir) / f"{base_name}.json"
    with open(json_path, 'w') as f:
        json.dump({
            "model": model_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "endpoint": ENDPOINT,
            "max_ctx": MAX_CTX,
            "results": results
        }, f, indent=2)
    log(f"✓ JSON: {json_path}")
    
    # CSV
    csv_path = Path(output_dir) / f"{base_name}.csv"
    with open(csv_path, 'w') as f:
        f.write("target_ctx,actual_ctx,pp_toks,tg_toks,pp_ms,tg_ms,wall_s\n")
        for r in results:
            f.write(f"{r['target_ctx']},{r['actual_ctx']},{r['pp_toks']},{r['tg_toks']},"
                   f"{r['pp_ms']},{r['tg_ms']},{r['wall_time_s']}\n")
    log(f"✓ CSV: {csv_path}")
    
    # Markdown
    md_path = Path(output_dir) / f"{base_name}.md"
    with open(md_path, 'w') as f:
        f.write(f"# Context Ladder — {model_name}\n\n")
        f.write(f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")
        f.write(f"**Endpoint:** {ENDPOINT}\n")
        f.write(f"**Max slot ctx:** {MAX_CTX:,}\n\n")
        f.write("| Context | PP tok/s | TG tok/s | PP ms | TG ms | Wall s |\n")
        f.write("|--------:|---------:|---------:|------:|------:|-------:|\n")
        for r in results:
            f.write(f"| {r['actual_ctx']:,} | {r['pp_toks']:.1f} | {r['tg_toks']:.1f} | "
                   f"{r['pp_ms']:.0f} | {r['tg_ms']:.0f} | {r['wall_time_s']:.1f} |\n")
    log(f"✓ Markdown: {md_path}")


def main():
    # Parse test points
    if os.getenv("TEST_POINTS"):
        test_points = [int(x.strip()) for x in os.environ["TEST_POINTS"].split(",")]
    else:
        test_points = [t for t in DEFAULT_TEST_POINTS if t <= MAX_CTX]
    
    print("=" * 70)
    print("Context Ladder Benchmark")
    print("=" * 70)
    log(f"Model:      {MODEL_NAME}")
    log(f"Endpoint:   {ENDPOINT}")
    log(f"SSH Host:   {SSH_HOST}")
    log(f"Max CTX:    {MAX_CTX:,}")
    log(f"Test points: {test_points}")
    log(f"Output:     {OUTPUT_DIR}")
    print("=" * 70)
    
    results = []
    
    for i, target_ctx in enumerate(test_points):
        result = test_context_point(ENDPOINT, SSH_HOST, target_ctx)
        if result:
            results.append(result)
        else:
            log(f"  Skipping {target_ctx:,}")
        
        log(f"  Progress: {i+1}/{len(test_points)}")
        time.sleep(3)  # cooldown between tests
    
    if results:
        print("=" * 70)
        log(f"Completed {len(results)}/{len(test_points)} tests")
        print("=" * 70)
        save_results(results, MODEL_NAME, OUTPUT_DIR)
    else:
        log("No results collected!")
        sys.exit(1)


if __name__ == "__main__":
    main()
