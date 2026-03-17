# Overview

**llmlab** is a lightweight local-LLM lab built around reproducible configurations, comparable benchmarks, and practical operator notes.

The project has three goals: keep runtime configs and results in one place, make quality/performance comparisons repeatable, and share a realistic setup others can run without lab-specific assumptions. One recurring theme is that apparently marginal consumer hardware can still be operationally useful when tuned carefully: for example, the current 3× RTX 3060 setup runs meaningful multi-GPU workloads even though two cards sit on PCIe Gen3 x4 links, and those links are not the dominant bottleneck under the validated vLLM PP=3 profile.

## What’s in this repo

The repository is organized around day-to-day operations and durable learning. Runbooks cover service lifecycle and standard checks, benchmarks document flags and interpretation, and experiments follow a consistent structure (goal → setup → commands → metrics → conclusion). For policy-level behavior, see `docs/thinking-policy.md`; for command-level control, see `docs/localbot.md`.

## Current focus (Feb 2026)

The current emphasis is Nemotron reasoning-profile tuning (A/B/C) and fair comparison discipline across two run classes: `reduced-thinking-balanced` and `non-thinking-speed`. The practical objective is clear operating guidance for speed-versus-quality tradeoffs in daily use.

## What is intentionally excluded

This repo stays public-safe by design. It does not include secrets, private infrastructure details, or user-specific paths.
