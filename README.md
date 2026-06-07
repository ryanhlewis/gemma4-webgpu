---
title: Gemma 4 WebGPU
sdk: gradio
app_file: app.py
python_version: "3.11"
models:
  - ryanhlewis/gemma-4-E2B-it-qat-q4_0-gguf-webgpu
  - gguf-org/gemma-4-e2b-it-gguf
tags:
  - webgpu
  - gguf
  - llama.cpp
  - wllama
  - gemma
---

# Gemma 4 WebGPU

Browser chat app for `ryanhlewis/gemma-4-E2B-it-qat-q4_0-gguf-webgpu`.

The Space serves only HTML, CSS, JavaScript, and Wllama WASM assets from the checked-in `dist/` build. The GGUF files are downloaded by the browser and inference runs locally through Wllama / llama.cpp WebGPU. The Python entrypoint is only a static file server with cross-origin isolation headers.

## Model Assets

The official Google repository contains:

| File | Size | Use |
| --- | ---: | --- |
| `gemma-4-E2B_q4_0-it.gguf` | 3.35 GB | language model |
| `mmproj-gemma-4-e2b-it-q4_0.gguf` | 336 MB | q4 vision projector |

Wllama supports split GGUF loading, so the app defaults to a public mirror containing split shards of the 3.35 GB model file plus a q4 vision projector. This keeps each browser-fetched LLM shard under the per-file browser/WASM limit while preserving the same language model weights.

## Local Development

```powershell
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:8030/
```

Cross-origin isolation headers are enabled in Vite and in `app.py` because Wllama uses browser features such as WebGPU, WebAssembly, OPFS, and SharedArrayBuffer when available.

## URL Overrides

The app accepts:

```text
?modelUrl=https://huggingface.co/<repo>/resolve/main/model-00001-of-00005.gguf
?mmprojUrl=https://huggingface.co/<repo>/resolve/main/mmproj-gemma-4-e2b-it-q4_0.gguf
?ctx=4096&threads=4&maxTokens=384
```

For split models, pass the first shard URL. Wllama discovers the remaining shards from the `-00001-of-00005.gguf` naming pattern.

## Producing Split Shards

Use the helper script after installing Docker and Hugging Face CLI auth:

```powershell
.\scripts\split-gemma-gguf.ps1
```

It downloads the official Google GGUF, uses the llama.cpp Docker image to run `llama-gguf-split`, and uploads the split shards plus the mmproj file to a public Hugging Face model repo.

## References

- Wllama supports direct browser inference, WebGPU, multimodal inputs, and split GGUF loading.
- The Hugging Face Space uses the free Gradio runtime as a static file server so it does not require Static Space build credits.
- `google/gemma-4-E2B-it-qat-q4_0-gguf` is Apache-2.0 licensed.
