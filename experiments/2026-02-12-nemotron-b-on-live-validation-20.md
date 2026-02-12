# Nemotron B_on live validation (20 prompts)

- Timestamp (UTC): 2026-02-12T18:49:46.699727Z
- Endpoint: `http://192.168.0.27:8080/v1/chat/completions`
- Model: `Nemotron-3-Nano-30B-A3B-IQ4_NL.gguf`
- Profile: **B_on** (`--reasoning-budget -1` + brief constrained reasoning prompt)

## Summary
- n_ok: **20**
- n_total: **20**
- avg_latency_s: **2.6**
- p50_latency_s: **2.784**
- p90_latency_s: **3.81**
- reasoning_tokens_total: **1549**
- visible_tokens_total: **303**
- reasoning_to_visible_ratio: **5.112**
- leak_count: **0**
- prompt_tokens_total: **1107**
- completion_tokens_total: **2940**
- output_tokens_per_s: **56.55**

## Per-prompt

| # | latency_s | rtok | vtok | ratio | leak |
|---:|---:|---:|---:|---:|:---:|
| 1 | 1.376 | 46 | 1 | 46.0 | false |
| 2 | 1.042 | 32 | 6 | 5.333 | false |
| 3 | 3.8 | 93 | 6 | 15.5 | false |
| 4 | 2.975 | 48 | 68 | 0.706 | false |
| 5 | 3.411 | 72 | 61 | 1.18 | false |
| 6 | 1.29 | 37 | 6 | 6.167 | false |
| 7 | 1.153 | 25 | 1 | 25.0 | false |
| 8 | 3.792 | 53 | 1 | 53.0 | false |
| 9 | 2.593 | 72 | 21 | 3.429 | false |
| 10 | 2.404 | 25 | 62 | 0.403 | false |
| 11 | 1.612 | 29 | 6 | 4.833 | false |
| 12 | 3.803 | 156 | 0 | 156.0 | false |
| 13 | 1.006 | 34 | 6 | 5.667 | false |
| 14 | 3.79 | 159 | 0 | 159.0 | false |
| 15 | 3.795 | 144 | 1 | 144.0 | false |
| 16 | 0.837 | 22 | 6 | 3.667 | false |
| 17 | 1.871 | 25 | 40 | 0.625 | false |
| 18 | 3.822 | 147 | 11 | 13.364 | false |
| 19 | 3.81 | 177 | 0 | 177.0 | false |
| 20 | 3.811 | 153 | 0 | 153.0 | false |
