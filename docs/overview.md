# Overview

**llmlab** is a lightweight local-LLM lab built around reproducible configurations, comparable benchmarks, and practical operator notes.

## Goals
- Keep runtime configs, benchmark outputs, and conclusions in one place.
- Make performance and quality comparisons repeatable.
- Share a minimal, realistic setup others can run without lab-specific assumptions.

## What’s in this repo
- **Runbooks** for starting/stopping services and standard checks.
- **Benchmarks** with explicit flags and interpretation notes.
- **Experiments** documented as goal → setup → commands → metrics → conclusion.
- **Reasoning policy** (`docs/thinking-policy.md`).
- **LocalBot command reference** (`docs/localbot.md`).

## Current focus (Feb 2026)
- Nemotron reasoning-profile tuning (A/B/C)
- profile-class separation for fair comparisons:
  - `reduced-thinking-balanced`
  - `non-thinking-speed`
- speed vs quality operating guidance for day-to-day use

## What is intentionally excluded
- secrets, credentials, and private infrastructure data
- user-specific hostnames/IPs/paths
