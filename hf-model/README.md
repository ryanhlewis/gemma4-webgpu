---
license: apache-2.0
base_model: google/gemma-4-E2B-it
tags:
  - gguf
  - webgpu
  - wllama
  - llama.cpp
  - gemma
---

# Gemma 4 E2B IT QAT Q4_0 GGUF WebGPU Shards

Browser-safe mirror of `google/gemma-4-E2B-it-qat-q4_0-gguf` for Wllama / llama.cpp WebGPU.

The language model GGUF is the upstream Google QAT Q4_0 file split with `llama-gguf-split --split-max-size 512M`. The split keeps every browser-fetched LLM shard below the practical per-file browser/WASM limit while preserving the original model weights. The matching mmproj file is included in this repo so the static app can load all required assets from one public model repository.

## Files

| File | Size | Use |
| --- | ---: | --- |
| `gemma-4-E2B_q4_0-it-00001-of-00005.gguf` | 43.3 MB | first split shard |
| `gemma-4-E2B_q4_0-it-00002-of-00005.gguf` | 1.93 GB | split shard |
| `gemma-4-E2B_q4_0-it-00003-of-00005.gguf` | 512 MB | split shard |
| `gemma-4-E2B_q4_0-it-00004-of-00005.gguf` | 505 MB | split shard |
| `gemma-4-E2B_q4_0-it-00005-of-00005.gguf` | 362 MB | split shard |
| `gemma-4-E2B-it-mmproj.gguf` | 987 MB | vision projector |

## Wllama URLs

Pass the first LLM shard URL. Wllama discovers the remaining shards from the filename pattern:

```text
https://huggingface.co/ryanhlewis/gemma-4-E2B-it-qat-q4_0-gguf-webgpu/resolve/main/gemma-4-E2B_q4_0-it-00001-of-00005.gguf
```

Use this mmproj URL for image input:

```text
https://huggingface.co/ryanhlewis/gemma-4-E2B-it-qat-q4_0-gguf-webgpu/resolve/main/gemma-4-E2B-it-mmproj.gguf
```

Static demo:

```text
https://huggingface.co/spaces/ryanhlewis/gemma4-webgpu
```
