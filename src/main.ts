import { LoggerWithoutDebug, Wllama } from "@wllama/wllama";
import type { ChatCompletionChunk, ChatCompletionMessage, ResultTimings } from "@wllama/wllama";
import wasmUrl from "@wllama/wllama/esm/wasm/wllama.wasm?url";
import "./style.css";

const DEFAULT_LLM_URL =
  "https://huggingface.co/ryanhlewis/gemma-4-E2B-it-qat-q4_0-gguf-webgpu/resolve/main/gemma-4-E2B_q4_0-it-00001-of-00005.gguf";
const DEFAULT_MMPROJ_URL =
  "https://huggingface.co/ryanhlewis/gemma-4-E2B-it-qat-q4_0-gguf-webgpu/resolve/main/gemma-4-E2B-it-mmproj.gguf";
const GOOGLE_SINGLE_GGUF =
  "https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/resolve/main/gemma-4-E2B_q4_0-it.gguf";
const TOTAL_DOWNLOAD_BYTES = 4_340_000_000;

type Role = "user" | "assistant";

type ChatEntry = {
  role: Role;
  content: string;
};

type LoadStats = {
  loaded: number;
  total: number;
  startedAt: number;
  doneAt?: number;
};

const params = new URLSearchParams(window.location.search);
const root = document.querySelector<HTMLDivElement>("#app");
if (!root) throw new Error("#app missing");
const app = root;

let wllama: Wllama | null = null;
let modelLoaded = false;
let loading = false;
let generating = false;
let loadStats: LoadStats = { loaded: 0, total: TOTAL_DOWNLOAD_BYTES, startedAt: 0 };
let chat: ChatEntry[] = [];
let imageFile: File | null = null;
let abortController: AbortController | null = null;
let lastTimings: ResultTimings | undefined;

const state = {
  llmUrl: params.get("modelUrl") || params.get("llmUrl") || DEFAULT_LLM_URL,
  mmprojUrl: params.get("mmprojUrl") || DEFAULT_MMPROJ_URL,
  context: Number(params.get("ctx") || 4096),
  threads: Number(params.get("threads") || 4),
  batch: Number(params.get("batch") || 512),
  gpuLayers: Number(params.get("gpuLayers") || 999),
  maxTokens: Number(params.get("maxTokens") || 384),
  temperature: Number(params.get("temperature") || 0.7),
  topP: Number(params.get("topP") || 0.95),
  seed: Number(params.get("seed") || -1),
  prompt:
    params.get("prompt") ||
    "Explain what makes Gemma 4 E2B useful for browser-local AI in three concise bullets.",
};

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KiB", "MiB", "GiB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function formatSeconds(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "0.0s";
  return `${seconds.toFixed(1)}s`;
}

function elapsedLoadSeconds(): number {
  if (!loadStats.startedAt) return 0;
  const end = loadStats.doneAt || performance.now();
  return (end - loadStats.startedAt) / 1000;
}

function loadPercent(): number {
  if (!loadStats.total) return 0;
  return Math.max(0, Math.min(100, (loadStats.loaded / loadStats.total) * 100));
}

function webgpuStatus(): string {
  if (!("gpu" in navigator)) return "unavailable";
  return modelLoaded && wllama?.isSupportWebGPU() ? "ready" : "available";
}

function setLog(message: string): void {
  currentLog = message;
  const el = document.querySelector<HTMLDivElement>("#log");
  if (el) el.textContent = message;
}

function appendLog(message: string): void {
  currentLog = `${currentLog ? `${currentLog}\n` : ""}${message}`;
  const el = document.querySelector<HTMLDivElement>("#log");
  if (!el) return;
  el.textContent = currentLog;
  el.scrollTop = el.scrollHeight;
}

function render(): void {
  const progress = loadPercent();
  const loadLabel = modelLoaded
    ? "Model ready"
    : loading
      ? `${formatBytes(loadStats.loaded)} / ${formatBytes(loadStats.total)}`
      : "Download model assets";
  const loadSub = loading
    ? `${progress.toFixed(1)}% | ${formatBytes(loadStats.loaded / Math.max(elapsedLoadSeconds(), 0.1))}/s | ${formatSeconds(elapsedLoadSeconds())}`
    : "4.34 GB";

  app.innerHTML = `
    <main class="app">
      <header class="topbar">
        <div class="brand">
          <div class="mark">G4</div>
          <div>
            <h1>Gemma 4 WebGPU</h1>
            <p class="subtle">GGUF chat running locally in your browser with Wllama and llama.cpp WebGPU.</p>
          </div>
        </div>
        <div class="actions">
          <button class="icon-button" id="clearButton" title="Clear chat">C</button>
          <button class="icon-button" id="cacheButton" title="Clear model cache">R</button>
        </div>
      </header>

      <section class="shell">
        <section class="panel chat-panel">
          <div class="status-strip">
            <div class="metric"><span>Status</span><strong>${modelLoaded ? "ready" : loading ? "loading" : "idle"}</strong></div>
            <div class="metric"><span>WebGPU</span><strong>${webgpuStatus()}</strong></div>
            <div class="metric"><span>Prompt</span><strong>${lastTimings ? `${lastTimings.prompt_per_second.toFixed(1)} tok/s` : "-"}</strong></div>
            <div class="metric"><span>Generate</span><strong>${lastTimings ? `${lastTimings.predicted_per_second.toFixed(1)} tok/s` : "-"}</strong></div>
          </div>

          <div class="messages">
            ${
              chat.length
                ? `<div class="message-list">${chat
                    .map((item) => `<div class="message ${item.role}">${escapeHtml(item.content)}</div>`)
                    .join("")}</div>`
                : `<div class="empty"><div><h2>Ask Gemma 4 locally</h2><p class="subtle">Download once, then chat from browser cache.</p></div></div>`
            }
          </div>

          <form class="composer" id="chatForm">
            <textarea id="promptInput" placeholder="Message Gemma">${escapeHtml(state.prompt)}</textarea>
            <div class="composer-row">
              <label class="file-chip">
                <span class="chip-button">Add image</span>
                <input id="imageInput" type="file" accept="image/*" />
                <span class="file-name">${imageFile ? escapeHtml(imageFile.name) : "Optional image input"}</span>
              </label>
              ${imageFile ? `<button class="button" type="button" id="removeImageButton">Remove image</button>` : ""}
              ${
                generating
                  ? `<button class="button danger" id="stopButton" type="button">Stop</button>`
                  : `<button class="button primary" type="submit" ${!modelLoaded ? "disabled" : ""}>Send</button>`
              }
            </div>
          </form>
        </section>

        <aside class="side">
          <section class="panel card">
            <h2>${loadLabel}</h2>
            <p class="subtle">${loadSub}</p>
            <div class="field">
              <div class="progress" aria-label="Download progress">
                <div class="bar" style="width: ${progress}%"></div>
              </div>
            </div>
            <div class="field">
              <button class="button primary" id="loadButton" ${loading || modelLoaded ? "disabled" : ""}>Start download</button>
            </div>
            <div class="pill-row">
              <span class="pill">LLM q4_0 split</span>
              <span class="pill">mmproj</span>
              <span class="pill">WebGPU layers</span>
            </div>
          </section>

          <section class="panel card">
            <h2>Generation</h2>
            <div class="field-grid">
              <div class="field">
                <label for="maxTokens">Max tokens</label>
                <input id="maxTokens" type="number" min="16" max="4096" step="16" value="${state.maxTokens}" />
              </div>
              <div class="field">
                <label for="temperature">Temperature</label>
                <input id="temperature" type="number" min="0" max="2" step="0.05" value="${state.temperature}" />
              </div>
            </div>
            <div class="field-grid">
              <div class="field">
                <label for="topP">Top-p</label>
                <input id="topP" type="number" min="0.01" max="1" step="0.01" value="${state.topP}" />
              </div>
              <div class="field">
                <label for="seed">Seed</label>
                <input id="seed" type="number" step="1" value="${state.seed}" />
              </div>
            </div>
          </section>

          <section class="panel card">
            <h2>Runtime</h2>
            <div class="field-grid">
              <div class="field">
                <label for="context">Context</label>
                <input id="context" type="number" min="512" max="32768" step="512" value="${state.context}" />
              </div>
              <div class="field">
                <label for="threads">Threads</label>
                <input id="threads" type="number" min="1" max="16" step="1" value="${state.threads}" />
              </div>
            </div>
            <div class="field-grid">
              <div class="field">
                <label for="batch">Batch</label>
                <input id="batch" type="number" min="64" max="2048" step="64" value="${state.batch}" />
              </div>
              <div class="field">
                <label for="gpuLayers">GPU layers</label>
                <input id="gpuLayers" type="number" min="0" max="999" step="1" value="${state.gpuLayers}" />
              </div>
            </div>
          </section>

          <section class="panel card">
            <h2>Model source</h2>
            <div class="field">
              <label for="llmUrl">LLM GGUF first shard</label>
              <input id="llmUrl" class="source" value="${escapeAttr(state.llmUrl)}" />
            </div>
            <div class="field">
              <label for="mmprojUrl">mmproj GGUF</label>
              <input id="mmprojUrl" class="source" value="${escapeAttr(state.mmprojUrl)}" />
            </div>
            <div class="field">
              <button class="button" id="googleSingleButton" type="button">Use Google single file</button>
            </div>
          </section>

          <section class="panel card">
            <h2>Console</h2>
            <div class="log" id="log">${escapeHtml(currentLog)}</div>
          </section>
        </aside>
      </section>
    </main>
  `;

  bindEvents();
  requestAnimationFrame(() => {
    const messages = document.querySelector<HTMLDivElement>(".messages");
    if (messages) messages.scrollTop = messages.scrollHeight;
  });
}

let currentLog = "Ready.";

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function escapeAttr(value: string): string {
  return escapeHtml(value).replaceAll("\n", " ");
}

function syncInputs(): void {
  state.prompt = getTextAreaValue("promptInput", state.prompt);
  state.llmUrl = getInputValue("llmUrl", state.llmUrl);
  state.mmprojUrl = getInputValue("mmprojUrl", state.mmprojUrl);
  state.context = getNumberValue("context", state.context);
  state.threads = getNumberValue("threads", state.threads);
  state.batch = getNumberValue("batch", state.batch);
  state.gpuLayers = getNumberValue("gpuLayers", state.gpuLayers);
  state.maxTokens = getNumberValue("maxTokens", state.maxTokens);
  state.temperature = getNumberValue("temperature", state.temperature);
  state.topP = getNumberValue("topP", state.topP);
  state.seed = getNumberValue("seed", state.seed);
}

function getInputValue(id: string, fallback: string): string {
  return document.querySelector<HTMLInputElement>(`#${id}`)?.value.trim() || fallback;
}

function getTextAreaValue(id: string, fallback: string): string {
  return document.querySelector<HTMLTextAreaElement>(`#${id}`)?.value.trim() || fallback;
}

function getNumberValue(id: string, fallback: number): number {
  const value = Number(document.querySelector<HTMLInputElement>(`#${id}`)?.value);
  return Number.isFinite(value) ? value : fallback;
}

function bindEvents(): void {
  document.querySelector<HTMLButtonElement>("#loadButton")?.addEventListener("click", () => {
    void loadModel();
  });

  document.querySelector<HTMLFormElement>("#chatForm")?.addEventListener("submit", (event) => {
    event.preventDefault();
    void sendMessage();
  });

  document.querySelector<HTMLButtonElement>("#stopButton")?.addEventListener("click", () => {
    abortController?.abort();
  });

  document.querySelector<HTMLInputElement>("#imageInput")?.addEventListener("change", (event) => {
    const files = (event.target as HTMLInputElement).files;
    imageFile = files?.[0] || null;
    render();
  });

  document.querySelector<HTMLButtonElement>("#removeImageButton")?.addEventListener("click", () => {
    imageFile = null;
    render();
  });

  document.querySelector<HTMLButtonElement>("#clearButton")?.addEventListener("click", () => {
    chat = [];
    lastTimings = undefined;
    render();
  });

  document.querySelector<HTMLButtonElement>("#cacheButton")?.addEventListener("click", () => {
    void clearCache();
  });

  document.querySelector<HTMLButtonElement>("#googleSingleButton")?.addEventListener("click", () => {
    state.llmUrl = GOOGLE_SINGLE_GGUF;
    modelLoaded = false;
    render();
    appendLog("Switched to the official single GGUF URL. Browsers may reject files over 2 GB; split shards are recommended.");
  });
}

async function loadModel(): Promise<void> {
  if (loading || modelLoaded) return;
  syncInputs();

  loading = true;
  loadStats = { loaded: 0, total: TOTAL_DOWNLOAD_BYTES, startedAt: performance.now() };
  currentLog = "Creating Wllama runtime...";
  render();

  try {
    wllama = new Wllama(
      { default: wasmUrl },
      {
        logger: LoggerWithoutDebug,
        suppressNativeLog: true,
        parallelDownloads: 5,
      }
    );
    wllama.setCompat(null);

    if (!wllama.isSupportWebGPU()) {
      appendLog("WebGPU is not available. Wllama may fall back or fail depending on this browser.");
    }

    appendLog(`Loading ${state.llmUrl}`);
    await wllama.loadModelFromUrl(
      {
        url: state.llmUrl,
        mmprojUrl: state.mmprojUrl || undefined,
      },
      {
        useCache: true,
        n_ctx: state.context,
        n_threads: state.threads,
        n_batch: state.batch,
        n_gpu_layers: state.gpuLayers,
        flash_attn: true,
        cache_type_k: "f16",
        cache_type_v: "f16",
        mmproj_offload: true,
        seed: state.seed >= 0 ? state.seed : Math.floor(Math.random() * 2_147_483_647),
        progressCallback: ({ loaded, total }) => {
          loadStats.loaded = loaded;
          loadStats.total = total || loadStats.total || TOTAL_DOWNLOAD_BYTES;
          currentLog = `Downloading model assets...\n${formatBytes(loadStats.loaded)} / ${formatBytes(loadStats.total)}`;
          render();
        },
      }
    );

    loadStats.doneAt = performance.now();
    modelLoaded = true;
    currentLog = [
      `Loaded in ${formatSeconds(elapsedLoadSeconds())}`,
      `libllama ${Wllama.getLibllamaVersion()}`,
      `threads ${wllama.getNumThreads()}${wllama.isMultithread() ? " multi-thread" : " single-thread"}`,
      `modalities image=${safeSupport("image")} audio=${safeSupport("audio")}`,
    ].join("\n");
  } catch (error) {
    currentLog = `Load failed:\n${errorToText(error)}`;
    modelLoaded = false;
    void wllama?.exit().catch(() => undefined);
    wllama = null;
  } finally {
    loading = false;
    render();
  }
}

function safeSupport(kind: "image" | "audio"): string {
  try {
    return wllama?.supportInputModality(kind) ? "yes" : "no";
  } catch {
    return "unknown";
  }
}

async function clearCache(): Promise<void> {
  if (!wllama) {
    wllama = new Wllama({ default: wasmUrl }, { logger: LoggerWithoutDebug, suppressNativeLog: true });
  }
  await wllama.cacheManager.clear();
  modelLoaded = false;
  currentLog = "Browser model cache cleared.";
  render();
}

async function sendMessage(): Promise<void> {
  if (!wllama || !modelLoaded || generating) return;
  syncInputs();
  if (!state.prompt) return;

  const userMessage = state.prompt;
  const imageData = imageFile ? await imageFile.arrayBuffer() : null;
  const content = imageData
    ? [
        { type: "text" as const, text: userMessage },
        { type: "image" as const, data: imageData },
      ]
    : userMessage;

  const history: ChatCompletionMessage[] = chat.map((entry) => ({
    role: entry.role,
    content: entry.content,
  }));

  chat.push({ role: "user", content: imageFile ? `${userMessage}\n[image: ${imageFile.name}]` : userMessage });
  chat.push({ role: "assistant", content: "" });
  state.prompt = "";
  generating = true;
  abortController = new AbortController();
  lastTimings = undefined;
  const startedAt = performance.now();
  render();

  try {
    await wllama.createChatCompletion({
      messages: [
        ...history,
        {
          role: "user",
          content,
        },
      ],
      stream: true,
      max_tokens: state.maxTokens,
      temperature: state.temperature,
      top_p: state.topP,
      cache_prompt: true,
      timings_per_token: true,
      abortSignal: abortController.signal,
      onData: (chunk) => {
        const delta = chunk.choices?.[0]?.delta?.content || "";
        lastTimings = chunk.timings || lastTimings;
        const last = chat[chat.length - 1];
        if (last?.role === "assistant") last.content += delta;
        const elapsed = (performance.now() - startedAt) / 1000;
        currentLog = [
          `Generating ${formatSeconds(elapsed)}`,
          lastTimings ? `Prompt ${lastTimings.prompt_per_second.toFixed(1)} tok/s` : "",
          lastTimings ? `Output ${lastTimings.predicted_per_second.toFixed(1)} tok/s` : "",
        ]
          .filter(Boolean)
          .join("\n");
        render();
      },
    });
    const elapsed = (performance.now() - startedAt) / 1000;
    currentLog = formatDoneLog(elapsed, lastTimings);
  } catch (error) {
    const last = chat[chat.length - 1];
    if (last?.role === "assistant" && !last.content) {
      last.content = `Error: ${errorToText(error)}`;
    }
    currentLog = `Generation stopped:\n${errorToText(error)}`;
  } finally {
    generating = false;
    abortController = null;
    render();
  }
}

function formatDoneLog(elapsed: number, timings?: ResultTimings): string {
  return [
    `Done in ${formatSeconds(elapsed)}`,
    timings ? `Prompt ${timings.prompt_n} tokens at ${timings.prompt_per_second.toFixed(1)} tok/s` : "",
    timings ? `Generated ${timings.predicted_n} tokens at ${timings.predicted_per_second.toFixed(1)} tok/s` : "",
  ]
    .filter(Boolean)
    .join("\n");
}

function errorToText(error: unknown): string {
  if (error instanceof Error) return error.stack || error.message;
  return String(error);
}

render();
