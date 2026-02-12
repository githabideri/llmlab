# Nemotron A/B/C — timing + token breakdown (2026-02-12)

Token columns:
- `prompt_tok`: API prompt tokens
- `completion_tok`: API completion tokens
- `reason_tok`: extracted reasoning tokens
- `visible_tok`: extracted visible-answer tokens
- `residual_tok`: `completion_tok - reason_tok - visible_tok` (tags/formatting/other)

## Run set: budget -1 (`--reasoning-budget -1`)

- **Set total runtime (all 15 steps):** 43.92s
- **Set token totals:** prompt 661, completion 2528, reasoning 1418, visible 1065

### A_baseline_on

| step | prompt | latency_s | prompt_tok | completion_tok | reason_tok | visible_tok | residual_tok |
|---|---|---:|---:|---:|---:|---:|---:|
| p1 | Hi (1 sentence) | 0.88 | 30 | 41 | 35 | 3 | 3 |
| p2 | Capital of Austria | 0.90 | 35 | 46 | 36 | 7 | 3 |
| p3 | REST vs GraphQL (3 bullets) | 4.87 | 36 | 289 | 199 | 87 | 3 |
| p4 | 17×24 brief steps | 5.08 | 37 | 300 | 250 | 48 | 2 |
| p5 | 4-step NAS photo backup | 5.08 | 39 | 300 | 53 | 245 | 2 |
| **TOTAL** | 5 steps | **16.81** | **177** | **976** | **573** | **390** | **13** |
| **AVG/ratio** | — | **3.36** | — | — | — | — | reason:visible **1.47:1** |

### B_brief_on

| step | prompt | latency_s | prompt_tok | completion_tok | reason_tok | visible_tok | residual_tok |
|---|---|---:|---:|---:|---:|---:|---:|
| p1 | Hi (1 sentence) | 0.75 | 50 | 36 | 31 | 2 | 3 |
| p2 | Capital of Austria | 0.94 | 55 | 47 | 37 | 7 | 3 |
| p3 | REST vs GraphQL (3 bullets) | 4.16 | 56 | 243 | 169 | 71 | 3 |
| p4 | 17×24 brief steps | 2.16 | 57 | 120 | 58 | 59 | 3 |
| p5 | 4-step NAS photo backup | 3.56 | 59 | 205 | 78 | 124 | 3 |
| **TOTAL** | 5 steps | **11.57** | **277** | **651** | **373** | **263** | **15** |
| **AVG/ratio** | — | **2.31** | — | — | — | — | reason:visible **1.42:1** |

### C_prompt_off

| step | prompt | latency_s | prompt_tok | completion_tok | reason_tok | visible_tok | residual_tok |
|---|---|---:|---:|---:|---:|---:|---:|
| p1 | Hi (1 sentence) | 0.87 | 36 | 45 | 40 | 2 | 3 |
| p2 | Capital of Austria | 0.91 | 41 | 47 | 37 | 7 | 3 |
| p3 | REST vs GraphQL (3 bullets) | 4.39 | 42 | 258 | 180 | 75 | 3 |
| p4 | 17×24 brief steps | 4.26 | 43 | 251 | 138 | 110 | 3 |
| p5 | 4-step NAS photo backup | 5.11 | 45 | 300 | 77 | 218 | 5 |
| **TOTAL** | 5 steps | **15.54** | **207** | **901** | **472** | **412** | **17** |
| **AVG/ratio** | — | **3.11** | — | — | — | — | reason:visible **1.15:1** |

## Run set: budget 0 (`--reasoning-budget 0`)

- **Set total runtime (all 15 steps):** 25.43s
- **Set token totals:** prompt 676, completion 1405, reasoning 0, visible 1372

### A_baseline_budget0

| step | prompt | latency_s | prompt_tok | completion_tok | reason_tok | visible_tok | residual_tok |
|---|---|---:|---:|---:|---:|---:|---:|
| p1 | Hi (1 sentence) | 0.49 | 31 | 15 | 0 | 13 | 2 |
| p2 | Capital of Austria | 0.39 | 36 | 16 | 0 | 14 | 2 |
| p3 | REST vs GraphQL (3 bullets) | 2.79 | 37 | 161 | 0 | 160 | 1 |
| p4 | 17×24 brief steps | 1.45 | 38 | 80 | 0 | 78 | 2 |
| p5 | 4-step NAS photo backup | 5.09 | 40 | 300 | 0 | 297 | 3 |
| **TOTAL** | 5 steps | **10.21** | **182** | **572** | **0** | **562** | **10** |
| **AVG/ratio** | — | **2.04** | — | — | — | — | reason:visible **0.00:1** |

### B_brief_budget0

| step | prompt | latency_s | prompt_tok | completion_tok | reason_tok | visible_tok | residual_tok |
|---|---|---:|---:|---:|---:|---:|---:|
| p1 | Hi (1 sentence) | 0.33 | 51 | 10 | 0 | 6 | 4 |
| p2 | Capital of Austria | 0.43 | 56 | 18 | 0 | 14 | 4 |
| p3 | REST vs GraphQL (3 bullets) | 1.64 | 57 | 90 | 0 | 89 | 1 |
| p4 | 17×24 brief steps | 1.02 | 58 | 52 | 0 | 50 | 2 |
| p5 | 4-step NAS photo backup | 2.94 | 60 | 168 | 0 | 166 | 2 |
| **TOTAL** | 5 steps | **6.36** | **282** | **338** | **0** | **325** | **13** |
| **AVG/ratio** | — | **1.27** | — | — | — | — | reason:visible **0.00:1** |

### C_prompt_off_budget0

| step | prompt | latency_s | prompt_tok | completion_tok | reason_tok | visible_tok | residual_tok |
|---|---|---:|---:|---:|---:|---:|---:|
| p1 | Hi (1 sentence) | 0.25 | 37 | 8 | 0 | 6 | 2 |
| p2 | Capital of Austria | 0.40 | 42 | 16 | 0 | 14 | 2 |
| p3 | REST vs GraphQL (3 bullets) | 1.51 | 43 | 83 | 0 | 82 | 1 |
| p4 | 17×24 brief steps | 1.97 | 44 | 111 | 0 | 108 | 3 |
| p5 | 4-step NAS photo backup | 4.73 | 46 | 277 | 0 | 275 | 2 |
| **TOTAL** | 5 steps | **8.86** | **212** | **495** | **0** | **485** | **10** |
| **AVG/ratio** | — | **1.77** | — | — | — | — | reason:visible **0.00:1** |

