# Nemotron A/B/C Executive Summary (important metrics only)

## TL;DR
- **Recommended default:** `B_on` (brief-thinking, reasoning enabled)
- **Fastest mode:** `B_off` (brief-thinking, reasoning budget 0)

## Cross-system comparison table (5 tasks per run)

| Run | Time total | Avg / task | Input tok | Output tok | Thinking tok | Visible tok | Output tok/s | Thinking tok/s | Visible tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_on | 16.81s | 3.36s | 177 | 976 | 573 | 390 | 58.06 | 34.09 | 23.20 |
| **B_on (recommended)** | **11.57s** | **2.31s** | 277 | 651 | 373 | 263 | 56.27 | 32.24 | 22.73 |
| C_on | 15.54s | 3.11s | 207 | 901 | 472 | 412 | 57.98 | 30.37 | 26.51 |
| A_off | 10.21s | 2.04s | 182 | 572 | 0 | 562 | 56.02 | 0.00 | 55.04 |
| **B_off (fastest)** | **6.36s** | **1.27s** | 282 | 338 | 0 | 325 | 53.14 | 0.00 | 51.10 |
| C_off | 8.86s | 1.77s | 212 | 495 | 0 | 485 | 55.87 | 0.00 | 54.74 |

## What matters
1. **Gen speed is similar** across runs (~53–58 output tok/s).
2. Runtime differences come mainly from **how many output tokens** are produced.
3. `B_on` is best balance (quality + controlled thinking + good speed).
4. `B_off` is fastest but has known output-cleanliness risk in off-mode class.

## Note on token accounting
- `Output tok` is API completion tokens.
- `Thinking tok` + `Visible tok` can be slightly lower than `Output tok` due to tag/formatting residual tokens.
