param(
  [string]$SourceRepo = "google/gemma-4-E2B-it-qat-q4_0-gguf",
  [string]$TargetRepo = "ryanhlewis/gemma-4-E2B-it-qat-q4_0-gguf-webgpu",
  [string]$ModelFile = "gemma-4-E2B_q4_0-it.gguf",
  [string]$MmprojFile = "gemma-4-E2B-it-mmproj.gguf",
  [string]$SplitMaxSize = "512M",
  [string]$WorkDir = "model-cache"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null

Write-Host "Downloading $SourceRepo / $ModelFile"
hf download $SourceRepo $ModelFile --local-dir $WorkDir

Write-Host "Downloading $SourceRepo / $MmprojFile"
hf download $SourceRepo $MmprojFile --local-dir $WorkDir

$dockerWork = (Resolve-Path $WorkDir).Path

Write-Host "Splitting model into $SplitMaxSize shards"
docker run --rm --entrypoint /app/llama-gguf-split -v "${dockerWork}:/work" ghcr.io/ggml-org/llama.cpp:full `
  --split-max-size $SplitMaxSize "/work/$ModelFile" "/work/gemma-4-E2B_q4_0-it"

Write-Host "Creating or updating Hugging Face model repo $TargetRepo"
hf repo create $TargetRepo --type model --yes

@"
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

Browser-safe split mirror of `google/gemma-4-E2B-it-qat-q4_0-gguf`.

The LLM GGUF is split with `llama-gguf-split --split-max-size $SplitMaxSize` for Wllama/WebGPU browser loading. The mirror also includes the matching mmproj file:

`gemma-4-E2B-it-mmproj.gguf`

Use the first shard URL in Wllama:

```text
gemma-4-E2B_q4_0-it-00001-of-00005.gguf
```
"@ | Set-Content -Path (Join-Path $WorkDir "README.md") -Encoding UTF8

Write-Host "Uploading split shards and README"
$env:HF_HUB_DISABLE_XET = "1"
hf upload $TargetRepo (Join-Path $WorkDir "README.md") README.md --repo-type model
Get-ChildItem $WorkDir -Filter "gemma-4-E2B_q4_0-it-*.gguf" | Sort-Object Name | ForEach-Object {
  hf upload $TargetRepo $_.FullName $_.Name --repo-type model
}
hf upload $TargetRepo (Join-Path $WorkDir $MmprojFile) $MmprojFile --repo-type model

Write-Host "Done."
