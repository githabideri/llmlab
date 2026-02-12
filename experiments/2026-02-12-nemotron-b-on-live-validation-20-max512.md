# Nemotron B_on live validation (20 prompts, max_tokens=512)

- Timestamp (UTC): 2026-02-12T18:54:10.533315+00:00
- Endpoint: `http://192.168.0.27:8080/v1/chat/completions`
- Model: `Nemotron-3-Nano-30B-A3B-IQ4_NL.gguf`
- Profile: B_on

## Summary
- n_ok: **20**
- n_total: **20**
- max_tokens: **512**
- avg_latency_s: **3.163**
- p50_latency_s: **2.808**
- p90_latency_s: **4.807**
- reasoning_tokens_total: **1691**
- visible_tokens_total: **510**
- reasoning_to_visible_ratio: **3.316**
- leak_count: **0**
- empty_visible_count: **1**
- prompt_tokens_total: **1107**
- completion_tokens_total: **3497**
- output_tokens_per_s: **55.27**

## Per-prompt

| # | latency_s | rtok | vtok | ratio | leak | empty |
|---:|---:|---:|---:|---:|:---:|:---:|
| 1 | 1.153 | 36 | 1 | 36.0 | false | false |
| 2 | 0.716 | 15 | 6 | 2.5 | false | false |
| 3 | 4.053 | 78 | 11 | 7.091 | false | false |
| 4 | 3.441 | 50 | 90 | 0.556 | false | false |
| 5 | 4.325 | 50 | 123 | 0.407 | false | false |
| 6 | 1.347 | 37 | 6 | 6.167 | false | false |
| 7 | 1.294 | 22 | 5 | 4.4 | false | false |
| 8 | 4.078 | 35 | 1 | 35.0 | false | false |
| 9 | 2.129 | 51 | 20 | 2.55 | false | false |
| 10 | 1.791 | 21 | 41 | 0.512 | false | false |
| 11 | 1.668 | 29 | 6 | 4.833 | false | false |
| 12 | 6.607 | 232 | 23 | 10.087 | false | false |
| 13 | 1.015 | 32 | 6 | 5.333 | false | false |
| 14 | 4.247 | 146 | 38 | 3.842 | false | false |
| 15 | 3.626 | 113 | 11 | 10.273 | false | false |
| 16 | 1.217 | 34 | 2 | 17.0 | false | false |
| 17 | 2.175 | 43 | 33 | 1.303 | false | false |
| 18 | 8.915 | 373 | 0 | 373.0 | false | true |
| 19 | 4.666 | 143 | 50 | 2.86 | false | false |
| 20 | 4.807 | 151 | 37 | 4.081 | false | false |
