import { LoggerWithoutDebug, Wllama } from "@wllama/wllama";
import type {
  ChatCompletionChunk,
  ChatCompletionMessage,
  ChatCompletionParams,
  ChatCompletionResponse,
  ResultTimings,
} from "@wllama/wllama";
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
  thinking?: string;
  thinkingOpen?: boolean;
};

type StreamParts = {
  answer: string;
  thinking: string;
  replaceAnswer: boolean;
  replaceThinking: boolean;
};

type ThinkParseState = {
  inThinking: boolean;
  tagBuffer: string;
};

type LoadPhase = "idle" | "preparing" | "downloading" | "initializing" | "ready" | "error";

type LoadStats = {
  loaded: number;
  total: number;
  startedAt: number;
  phaseStartedAt: number;
  doneAt?: number;
};

const params = new URLSearchParams(window.location.search);
const root = document.querySelector<HTMLDivElement>("#app");
if (!root) throw new Error("#app missing");
const app = root;

let wllama: Wllama | null = null;
let modelLoaded = false;
let loading = false;
let loadPhase: LoadPhase = "idle";
let generating = false;
let loadStats: LoadStats = { loaded: 0, total: TOTAL_DOWNLOAD_BYTES, startedAt: 0, phaseStartedAt: 0 };
let chat: ChatEntry[] = [];
let imageFile: File | null = null;
let abortController: AbortController | null = null;
let lastTimings: ResultTimings | undefined;
let settingsOpen = false;

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
  thinking: params.get("thinking") === "hidden" ? "hidden" : "collapsed",
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

function elapsedPhaseSeconds(): number {
  if (!loadStats.phaseStartedAt) return 0;
  return (performance.now() - loadStats.phaseStartedAt) / 1000;
}

function setLoadPhase(phase: LoadPhase): void {
  if (loadPhase === phase) return;
  loadPhase = phase;
  loadStats.phaseStartedAt = performance.now();
}

function loadPercent(): number {
  if (!loadStats.total) return 0;
  return Math.max(0, Math.min(100, (loadStats.loaded / loadStats.total) * 100));
}

function statusLabel(): string {
  if (modelLoaded) return "ready";
  if (!loading) return loadPhase === "error" ? "error" : "idle";
  if (loadPhase === "downloading") return "downloading";
  if (loadPhase === "initializing") return "initializing";
  return "preparing";
}

function loadTitle(): string {
  if (modelLoaded) return "Model ready";
  if (loadPhase === "preparing") return "Preparing runtime";
  if (loadPhase === "downloading") return "Downloading model assets";
  if (loadPhase === "initializing") return "Loading model into WebGPU";
  if (loadPhase === "error") return "Load failed";
  return "Download model assets";
}

function loadDetail(progress: number): string {
  if (modelLoaded) return `Ready in ${formatSeconds(elapsedLoadSeconds())}`;
  if (loading && loadPhase === "preparing") {
    return `Starting browser runtime | ${formatSeconds(elapsedPhaseSeconds())}`;
  }
  if (loading && loadPhase === "downloading") {
    return `${progress.toFixed(1)}% | ${formatBytes(loadStats.loaded)} / ${formatBytes(loadStats.total)} | ${formatBytes(
      loadStats.loaded / Math.max(elapsedLoadSeconds(), 0.1)
    )}/s`;
  }
  if (loading && loadPhase === "initializing") {
    return `Download complete | Initializing ${formatSeconds(elapsedPhaseSeconds())}`;
  }
  return "4.34 GB";
}

function webgpuStatus(): string {
  if (!("gpu" in navigator)) return "unavailable";
  return modelLoaded && wllama?.isSupportWebGPU() ? "ready" : "available";
}

function iconSvg(name: "settings" | "newChat"): string {
  if (name === "settings") {
    return `
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M12 15.5A3.5 3.5 0 1 0 12 8a3.5 3.5 0 0 0 0 7.5Z"></path>
        <path d="M19.4 15a1.8 1.8 0 0 0 .36 1.98l.05.05a2.1 2.1 0 1 1-2.97 2.97l-.05-.05a1.8 1.8 0 0 0-1.98-.36 1.8 1.8 0 0 0-1.09 1.65V21a2.1 2.1 0 1 1-4.2 0v-.07a1.8 1.8 0 0 0-1.18-1.67 1.8 1.8 0 0 0-1.98.36l-.05.05a2.1 2.1 0 1 1-2.97-2.97l.05-.05a1.8 1.8 0 0 0 .36-1.98 1.8 1.8 0 0 0-1.65-1.09H3a2.1 2.1 0 1 1 0-4.2h.07a1.8 1.8 0 0 0 1.67-1.18 1.8 1.8 0 0 0-.36-1.98l-.05-.05A2.1 2.1 0 1 1 7.3 3.3l.05.05a1.8 1.8 0 0 0 1.98.36H9.4A1.8 1.8 0 0 0 10.5 2h3a1.8 1.8 0 0 0 1.09 1.65 1.8 1.8 0 0 0 1.98-.36l.05-.05a2.1 2.1 0 1 1 2.97 2.97l-.05.05a1.8 1.8 0 0 0-.36 1.98v.07A1.8 1.8 0 0 0 21 9.5h.07a2.1 2.1 0 1 1 0 4.2H21a1.8 1.8 0 0 0-1.6 1.3Z"></path>
      </svg>
    `;
  }
  return `
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
      <path d="M18.4 2.6a2.1 2.1 0 0 1 3 3L12 15l-4 1 1-4 9.4-9.4Z"></path>
    </svg>
  `;
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

function renderMessage(item: ChatEntry, index: number): string {
  if (item.role === "user") {
    return `<div class="message user">${escapeHtml(item.content)}</div>`;
  }

  const thinking = item.thinking?.trim() || "";
  const shouldShowThinking = state.thinking === "collapsed" && thinking.length > 0;
  const arrow = item.thinkingOpen ? "^" : "v";
  const thinkingBlock = shouldShowThinking
    ? `<div class="thinking-block">
        <button class="thinking-toggle" type="button" data-thinking-index="${index}">
          <span>Thinking...</span>
          <span class="thinking-arrow">${arrow}</span>
        </button>
        ${
          item.thinkingOpen
            ? `<div class="thinking-content">${escapeHtml(thinking)}</div>`
            : ""
        }
      </div>`
    : "";
  const answer = item.content.trim()
    ? escapeHtml(item.content)
    : `<span class="subtle">Waiting for response...</span>`;

  return `<div class="message assistant">${thinkingBlock}<div class="answer-text">${answer}</div></div>`;
}

function render(): void {
  const progress = loadPercent();
  const progressClass = loadPhase === "initializing" ? "progress indeterminate" : "progress";
  const barWidth = loadPhase === "preparing" && loading ? 8 : loadPhase === "initializing" ? 100 : progress;

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
          <button class="icon-button" id="settingsButton" title="Settings" aria-label="Settings">${iconSvg("settings")}</button>
          <button class="icon-button" id="clearButton" title="New chat" aria-label="New chat">${iconSvg("newChat")}</button>
          ${
            settingsOpen
              ? `<div class="settings-popover panel">
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
                </div>`
              : ""
          }
        </div>
      </header>

      <section class="shell">
        <section class="panel chat-panel">
          <div class="status-strip">
            <div class="metric"><span>Status</span><strong>${statusLabel()}</strong></div>
            <div class="metric"><span>WebGPU</span><strong>${webgpuStatus()}</strong></div>
            <div class="metric"><span>Prompt</span><strong>${lastTimings ? `${lastTimings.prompt_per_second.toFixed(1)} tok/s` : "-"}</strong></div>
            <div class="metric"><span>Generate</span><strong>${lastTimings ? `${lastTimings.predicted_per_second.toFixed(1)} tok/s` : "-"}</strong></div>
          </div>

          <div class="messages">
            ${
              chat.length
                ? `<div class="message-list">${chat
                    .map((item, index) => renderMessage(item, index))
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
            <h2>${loadTitle()}</h2>
            <p class="subtle">${loadDetail(progress)}</p>
            <div class="field">
              <div class="${progressClass}" aria-label="Model loading progress">
                <div class="bar" style="width: ${barWidth}%"></div>
              </div>
            </div>
            ${!loading && !modelLoaded ? `<div class="field"><button class="button primary" id="loadButton">Start download</button></div>` : ""}
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
            <div class="field">
              <label for="thinking">Thinking</label>
              <select id="thinking">
                <option value="collapsed" ${state.thinking === "collapsed" ? "selected" : ""}>Show collapsed</option>
                <option value="hidden" ${state.thinking === "hidden" ? "selected" : ""}>Hide</option>
              </select>
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
  state.thinking = getInputValue("thinking", state.thinking) === "hidden" ? "hidden" : "collapsed";
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

  document.querySelector<HTMLButtonElement>("#settingsButton")?.addEventListener("click", () => {
    syncInputs();
    settingsOpen = !settingsOpen;
    render();
  });

  document.querySelector<HTMLSelectElement>("#thinking")?.addEventListener("change", () => {
    syncInputs();
    render();
  });

  document.querySelectorAll<HTMLButtonElement>("[data-thinking-index]").forEach((button) => {
    button.addEventListener("click", () => {
      const index = Number(button.dataset.thinkingIndex);
      const entry = chat[index];
      if (!entry || entry.role !== "assistant") return;
      entry.thinkingOpen = !entry.thinkingOpen;
      render();
    });
  });

  document.querySelector<HTMLButtonElement>("#googleSingleButton")?.addEventListener("click", () => {
    syncInputs();
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
  const startedAt = performance.now();
  loadStats = { loaded: 0, total: TOTAL_DOWNLOAD_BYTES, startedAt, phaseStartedAt: startedAt };
  setLoadPhase("preparing");
  currentLog = "Preparing browser runtime...";
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

    appendLog(`Model source: ${state.llmUrl}`);
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
          if (loadStats.total && loadStats.loaded >= loadStats.total) {
            setLoadPhase("initializing");
            currentLog = [
              "Download complete.",
              `Initializing WebGPU model for ${formatSeconds(elapsedPhaseSeconds())}`,
              "This step loads weights and builds GPU resources.",
            ].join("\n");
          } else {
            setLoadPhase("downloading");
            currentLog = [
              "Downloading model assets...",
              `${formatBytes(loadStats.loaded)} / ${formatBytes(loadStats.total)}`,
              `${formatBytes(loadStats.loaded / Math.max(elapsedLoadSeconds(), 0.1))}/s`,
            ].join("\n");
          }
          render();
        },
      }
    );

    loadStats.doneAt = performance.now();
    modelLoaded = true;
    setLoadPhase("ready");
    currentLog = [
      `Loaded in ${formatSeconds(elapsedLoadSeconds())}`,
      `libllama ${Wllama.getLibllamaVersion()}`,
      `threads ${wllama.getNumThreads()}${wllama.isMultithread() ? " multi-thread" : " single-thread"}`,
      `modalities image=${safeSupport("image")} audio=${safeSupport("audio")}`,
    ].join("\n");
  } catch (error) {
    setLoadPhase("error");
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

  const history: ChatCompletionMessage[] = chat
    .filter((entry) => entry.content.trim().length > 0)
    .map((entry) => ({
      role: entry.role,
      content: entry.content,
    }));

  chat.push({ role: "user", content: imageFile ? `${userMessage}\n[image: ${imageFile.name}]` : userMessage });
  const assistantIndex = chat.length;
  chat.push({ role: "assistant", content: "" });
  state.prompt = "";
  generating = true;
  abortController = new AbortController();
  lastTimings = undefined;
  const startedAt = performance.now();
  const thinkParser: ThinkParseState = { inThinking: false, tagBuffer: "" };
  render();

  try {
    const userTurn = {
      role: "user",
      content,
    } as ChatCompletionMessage;
    const thinkingInstruction =
      state.thinking === "hidden"
        ? ([
            {
              role: "system",
              content: "Answer directly. Do not include thinking, reasoning traces, or <think> sections.",
            },
          ] as ChatCompletionMessage[])
        : [];
    const request: ChatCompletionParams = {
      messages: [
        ...thinkingInstruction,
        ...history,
        userTurn,
      ],
      max_tokens: state.maxTokens,
      temperature: state.temperature,
      top_p: state.topP,
      seed: state.seed >= 0 ? state.seed : undefined,
      cache_prompt: true,
      timings_per_token: true,
    };

    await wllama.createChatCompletion({
      ...request,
      stream: true,
      abortSignal: abortController.signal,
      onData: (chunk: ChatCompletionChunk) => {
        const delta = extractStreamParts(chunk, thinkParser);
        lastTimings = chunk.timings || lastTimings;
        const assistant = chat[assistantIndex];
        if (assistant?.role === "assistant") {
          if (delta.answer) {
            assistant.content = delta.replaceAnswer ? delta.answer : assistant.content + delta.answer;
          }
          if (delta.thinking) {
            assistant.thinking = delta.replaceThinking ? delta.thinking : `${assistant.thinking || ""}${delta.thinking}`;
          }
        }
        const elapsed = (performance.now() - startedAt) / 1000;
        currentLog = [
          `Generating ${formatSeconds(elapsed)}`,
          assistant?.content ? `Answer ${assistant.content.length} chars` : "Waiting for answer text...",
          assistant?.thinking ? `Thinking ${assistant.thinking.length} chars` : "",
          lastTimings ? `Prompt ${lastTimings.prompt_per_second.toFixed(1)} tok/s` : "",
          lastTimings ? `Output ${lastTimings.predicted_per_second.toFixed(1)} tok/s` : "",
        ]
          .filter(Boolean)
          .join("\n");
        render();
      },
    });
    const assistant = chat[assistantIndex];
    const flushed = flushThinkParser(thinkParser);
    if (assistant?.role === "assistant") {
      if (flushed.answer) assistant.content += flushed.answer;
      if (flushed.thinking) assistant.thinking = `${assistant.thinking || ""}${flushed.thinking}`;
    }
    if (assistant?.role === "assistant" && !assistant.content.trim()) {
      currentLog = "Stream returned timings but no visible text. Fetching final response...";
      render();
      const response = await wllama.createChatCompletion({
        ...request,
        stream: false,
        abortSignal: abortController.signal,
      });
      const finalParts = extractFinalParts(response);
      assistant.content = finalParts.answer;
      assistant.thinking = finalParts.thinking || assistant.thinking;
    }
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

function textFromValue(value: unknown): string {
  if (typeof value === "string") return value;
  if (value == null) return "";
  if (Array.isArray(value)) {
    return value
      .map((item) => {
        if (typeof item === "string") return item;
        if (item && typeof item === "object") {
          const typed = item as { text?: unknown; content?: unknown };
          return textFromValue(typed.text || typed.content);
        }
        return "";
      })
      .join("");
  }
  return "";
}

function splitTaggedThinking(text: string, state: ThinkParseState): { answer: string; thinking: string } {
  let answer = "";
  let thinking = "";
  const tags = ["<think>", "</think>", "<thinking>", "</thinking>"];

  const pushText = (value: string) => {
    if (state.inThinking) {
      thinking += value;
    } else {
      answer += value;
    }
  };

  for (const char of text) {
    if (state.tagBuffer) {
      state.tagBuffer += char;
      const lower = state.tagBuffer.toLowerCase();
      if (lower === "<think>" || lower === "<thinking>") {
        state.inThinking = true;
        state.tagBuffer = "";
        continue;
      }
      if (lower === "</think>" || lower === "</thinking>") {
        state.inThinking = false;
        state.tagBuffer = "";
        continue;
      }
      if (tags.some((tag) => tag.startsWith(lower))) {
        continue;
      }
      pushText(state.tagBuffer);
      state.tagBuffer = "";
      continue;
    }

    if (char === "<") {
      state.tagBuffer = char;
      continue;
    }
    pushText(char);
  }

  return { answer, thinking };
}

function flushThinkParser(state: ThinkParseState): { answer: string; thinking: string } {
  if (!state.tagBuffer) return { answer: "", thinking: "" };
  const pending = state.tagBuffer;
  state.tagBuffer = "";
  return state.inThinking ? { answer: "", thinking: pending } : { answer: pending, thinking: "" };
}

function splitCompleteTaggedThinking(text: string): { answer: string; thinking: string } {
  const parser: ThinkParseState = { inThinking: false, tagBuffer: "" };
  const parts = splitTaggedThinking(text, parser);
  const flushed = flushThinkParser(parser);
  return {
    answer: parts.answer + flushed.answer,
    thinking: parts.thinking + flushed.thinking,
  };
}

function extractStreamParts(chunk: unknown, state: ThinkParseState): StreamParts {
  const data = chunk as Record<string, unknown>;
  const choices = Array.isArray(data.choices) ? (data.choices as Array<Record<string, unknown>>) : [];
  const choice = choices[0] || {};
  const delta = (choice.delta && typeof choice.delta === "object" ? choice.delta : {}) as Record<string, unknown>;
  const message = (choice.message && typeof choice.message === "object" ? choice.message : {}) as Record<string, unknown>;
  const token = (data.token && typeof data.token === "object" ? data.token : {}) as Record<string, unknown>;

  const explicitThinking =
    textFromValue(delta.reasoning_content) ||
    textFromValue(delta.thinking) ||
    textFromValue(message.reasoning_content) ||
    textFromValue(message.thinking) ||
    textFromValue(data.reasoning_content) ||
    textFromValue(data.thinking);

  const appendText =
    textFromValue(delta.content) ||
    textFromValue(delta.text) ||
    textFromValue(choice.text) ||
    textFromValue(data.content) ||
    textFromValue(data.response) ||
    textFromValue(data.text) ||
    textFromValue(token.text);

  const replacementText = textFromValue(message.content) || textFromValue(data.message);
  const appendParts = splitTaggedThinking(appendText, state);
  const replacementParts = replacementText ? splitCompleteTaggedThinking(replacementText) : { answer: "", thinking: "" };

  return {
    answer: appendParts.answer || replacementParts.answer,
    thinking: explicitThinking || appendParts.thinking || replacementParts.thinking,
    replaceAnswer: Boolean(replacementParts.answer),
    replaceThinking: Boolean(explicitThinking || replacementParts.thinking),
  };
}

function extractFinalParts(response: ChatCompletionResponse): { answer: string; thinking: string } {
  const message = response.choices?.[0]?.message as unknown as Record<string, unknown> | undefined;
  const content = textFromValue(message?.content);
  const explicitThinking = textFromValue(message?.reasoning_content) || textFromValue(message?.thinking);
  const parsed = splitCompleteTaggedThinking(content);
  return {
    answer: parsed.answer,
    thinking: explicitThinking || parsed.thinking,
  };
}

render();
