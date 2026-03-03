#!/usr/bin/env python3
"""
Context Ladder Benchmark Script

Tests model performance across various context lengths.
Captures PP/TG tok/s, VRAM usage, CPU/RAM usage, and timing metrics.

Usage:
    ./run_context_ladder.py [--config path/to/.env]
"""

import json
import subprocess
import time
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Load .env if python-dotenv available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Using environment variables only.")

# Configuration from environment
ENDPOINT = os.getenv("LLAMA_ENDPOINT", "http://192.168.0.27:8080/v1/completions")
SSH_HOST = os.getenv("LLAMA_SSH_HOST", "llama-cpp")
MODEL_NAME = os.getenv("MODEL_NAME", "unknown-model")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "results")

# Test points (in tokens)
DEFAULT_TEST_POINTS = [
    64, 3200, 6400, 9600, 12800, 19200, 24000,
    32000, 64000, 95000, 128000, 192000, 250000
]


def get_process_stats(ssh_host: str, process_name: str = "llama-server") -> Dict:
    """Get CPU and RAM usage of llama-server process via SSH."""
    cmd = f"ssh {ssh_host} \"ps aux | grep {process_name} | grep -v grep | awk '{{print $3,$4,$5,$6}}'\""
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            cpu, mem_pct, vsz, rss = result.stdout.strip().split()
            return {
                "cpu_percent": float(cpu),
                "mem_percent": float(mem_pct),
                "vsz_kb": int(vsz),
                "rss_kb": int(rss),
                "rss_mb": round(int(rss) / 1024, 1)
            }
    except Exception as e:
        print(f"Warning: Could not get process stats: {e}")
    
    return {}


def get_gpu_stats(ssh_host: str) -> List[Dict]:
    """Get GPU memory usage via nvidia-smi over SSH."""
    cmd = f"ssh {ssh_host} \"nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits\""
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    idx, used, total, util, power = line.split(',')
                    gpus.append({
                        "index": int(idx.strip()),
                        "mem_used_mb": int(used.strip()),
                        "mem_total_mb": int(total.strip()),
                        "util_percent": int(util.strip()),
                        "power_watts": float(power.strip())
                    })
            return gpus
    except Exception as e:
        print(f"Warning: Could not get GPU stats: {e}")
    
    return []


def run_inference(endpoint: str, prompt: str, max_tokens: int = 128) -> Optional[Dict]:
    """Run inference request and return timing/token data."""
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "cache_prompt": False
    }
    
    try:
        proc = subprocess.Popen(
            ["curl", "-s", "-X", "POST", endpoint,
             "-H", "Content-Type: application/json",
             "-d", "@-",
             "-m", "900"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = proc.communicate(input=json.dumps(payload), timeout=920)
        
        if proc.returncode != 0:
            print(f"  Curl failed: {stderr[:200]}")
            return None
        
        return json.loads(stdout)
        
    except Exception as e:
        print(f"  Request failed: {e}")
        return None


def test_context_point(
    endpoint: str,
    ssh_host: str,
    target_tokens: int,
    process_name: str = "llama-server"
) -> Optional[Dict]:
    """Test a single context point and gather all metrics."""
    
    print(f"Testing ~{target_tokens} tokens context...", flush=True)
    
    # Build prompt
    if target_tokens <= 64:
        prompt = "What is 2+2? Answer briefly:"
    else:
        padding = "The quick brown fox jumps over the lazy dog. " * int(target_tokens / 10)
        prompt = f"{padding}\n\nQuestion: What is the capital of France? Answer briefly:"
    
    # Get baseline stats
    pre_cpu_ram = get_process_stats(ssh_host, process_name)
    pre_gpu = get_gpu_stats(ssh_host)
    
    # Run inference
    start_time = time.time()
    data = run_inference(endpoint, prompt)
    end_time = time.time()
    
    if not data:
        return None
    
    # Get post-inference stats
    post_cpu_ram = get_process_stats(ssh_host, process_name)
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
    
    result = {
        "target_ctx": target_tokens,
        "actual_ctx": pp_tokens,
        "pp_toks": pp_toks,
        "tg_toks": tg_toks,
        "pp_ms": pp_ms,
        "tg_ms": tg_ms,
        "total_time_s": round(end_time - start_time, 2),
        "cpu_ram_pre": pre_cpu_ram,
        "cpu_ram_post": post_cpu_ram,
        "gpu_pre": pre_gpu,
        "gpu_post": post_gpu,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    
    print(f"  {pp_tokens:>6} ctx: PP={pp_toks:>7.2f} tok/s, TG={tg_toks:>6.2f} tok/s "
          f"(CPU: {post_cpu_ram.get('cpu_percent', 0):.1f}%, RAM: {post_cpu_ram.get('rss_mb', 0)} MB)")
    
    return result


def save_results(results: List[Dict], model_name: str, output_dir: str):
    """Save results in multiple formats."""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    base_name = f"{timestamp}-{model_name.replace('/', '-')}"
    
    # JSON (complete data)
    json_path = Path(output_dir) / f"{base_name}.json"
    with open(json_path, 'w') as f:
        json.dump({
            "model": model_name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "results": results
        }, f, indent=2)
    print(f"\n✓ JSON saved: {json_path}")
    
    # CSV (spreadsheet-friendly)
    csv_path = Path(output_dir) / f"{base_name}.csv"
    with open(csv_path, 'w') as f:
        f.write("context,pp_toks,tg_toks,pp_ms,tg_ms,cpu_pct,ram_mb\n")
        for r in results:
            cpu = r.get('cpu_ram_post', {}).get('cpu_percent', 0)
            ram = r.get('cpu_ram_post', {}).get('rss_mb', 0)
            f.write(f"{r['actual_ctx']},{r['pp_toks']},{r['tg_toks']},"
                   f"{r['pp_ms']},{r['tg_ms']},{cpu},{ram}\n")
    print(f"✓ CSV saved: {csv_path}")
    
    # Markdown (human-readable)
    md_path = Path(output_dir) / f"{base_name}.md"
    with open(md_path, 'w') as f:
        f.write(f"# Context Ladder — {model_name}\n\n")
        f.write(f"**Date:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write("| Context | PP tok/s | TG tok/s | CPU % | RAM MB |\n")
        f.write("|--------:|---------:|---------:|------:|-------:|\n")
        for r in results:
            cpu = r.get('cpu_ram_post', {}).get('cpu_percent', 0)
            ram = r.get('cpu_ram_post', {}).get('rss_mb', 0)
            f.write(f"| {r['actual_ctx']:,} | {r['pp_toks']:.2f} | {r['tg_toks']:.2f} | "
                   f"{cpu:.1f} | {ram:.0f} |\n")
    print(f"✓ Markdown saved: {md_path}")


def main():
    print("=" * 60)
    print("Context Ladder Benchmark")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Endpoint: {ENDPOINT}")
    print(f"SSH Host: {SSH_HOST}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)
    print()
    
    results = []
    
    for target_ctx in DEFAULT_TEST_POINTS:
        result = test_context_point(ENDPOINT, SSH_HOST, target_ctx)
        if result:
            results.append(result)
        else:
            print(f"  Skipping {target_ctx} due to error")
        
        # Small delay between tests
        time.sleep(2)
    
    if results:
        print("\n" + "=" * 60)
        print(f"Completed {len(results)}/{len(DEFAULT_TEST_POINTS)} tests")
        print("=" * 60)
        save_results(results, MODEL_NAME, OUTPUT_DIR)
    else:
        print("\nNo results collected!")
        sys.exit(1)


if __name__ == "__main__":
    main()
