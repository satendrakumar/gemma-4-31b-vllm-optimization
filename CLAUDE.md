# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A writing project, not an application. The main artifact is `blog.md`: a blog post about optimizing **Gemma-4-31B-it** inference with **vLLM** on a single **NVIDIA RTX PRO 6000** (Blackwell, 96 GB) running in GKE.

`main.py` and `pyproject.toml` are an untouched `uv init` scaffold (Python 3.13, no dependencies). `README.md` is empty. There is no build, lint, or test setup.

## Structure of `blog.md`

1. Opening that introduces Gemma 4 from the [technical report](https://arxiv.org/abs/2607.02770): thinking mode and benchmark table, MTP drafter, attention design, tool use, then "the catch" (slow out of the box). **Don't say or imply the author's team uses Gemini.** Gemini appears only as an outside reference point (deprecation news, pricing, accuracy).
2. Why an open model instead of a managed API (no forced retirement, fine-tuning, privacy, cost), why Gemma-4-31B-it over other open models (Gemma-4-26B-A4B-it, Qwen-3.6-35B-A3B, Qwen3.8-27B), and a detailed cost breakdown
3. "The results at a glance": summary table covering every step
4. Setup: workload profile, baseline `deploy.yaml`, baseline flag table, `vllm bench serve` command, baseline results
5. Steps 1–7, each with "What it does" / "Why it helps", a `diff` of the flag change, and a before/after table
6. Final `vllm serve` config, overall before/after table, key takeaways

## Conventions to keep consistent

- **Benchmark workload:** 16,384 input tokens, 512 output tokens, concurrency 8, 100 prompts. All numbers come from the author's own benchmark runs. Don't invent or "round up" results. Percentages are relative to the previous step, except in the final table, which compares against the baseline.
- **Results tables:** every table shows Throughput, Mean TTFT, Mean TPOT, and Mean ITL. If a metric moves unexpectedly, explain it next to the table. For example, mean ITL rises with speculative decoding because tokens arrive in multi-token chunks.
- **Cost math:** RTX PRO 6000 costs ~$3,000/month flat. Gemini-2.5-Flash costs $0.30/$2.50 per 1M input/output tokens; Gemini-3.5-Flash costs $1.50/$9.00. Monthly volume = throughput × 2,592,000 s (30 days at 24/7 full load). If throughput numbers change, recompute the cost section, the opening cost table, and the break-even caveat (~12% vs. 3.5 Flash, ~57% vs. 2.5 Flash) together.
- **Placeholders the author still needs to fill:** the commented-out eval score table (`xx%`) in the model-comparison section, and the `gcloud ... node-pools create` command.
- **Dollar signs:** write `\$` in prose and tables, because some Markdown previews treat a pair of `$` as LaTeX math.
- **Style:** short, scannable paragraphs; tables over dense prose; one or two sentences per parameter explanation. The author has asked for less verbose bullets before.
- **Accuracy:** flag claims that look technically wrong (for example, FA4 is a Blackwell feature, not a Hopper/H200 one; Gemini Flash does support Vertex AI tuning) instead of repeating them.
