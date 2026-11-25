// Imagine: type a sentence, and every pause paints it on your own GPU.
//
// This file holds the page's state and draws it into the DOM in index.html.
// State lives in a few plain variables; after changing them, call the matching
// render function (or render() for everything).

import { Painter, hasWebGPU } from "./engine.js";
import { OPENING, SUGGESTIONS, SURPRISES } from "./prompts.js";

// Where minit2i.onnx, minit2i.onnx.data and vocab.json are downloaded from.
// Point this at your own copy (for example "./model/") to serve the model yourself.
const MODEL_BASE =
  "https://huggingface.co/cfahlgren1/MiniT2I-B-16-MF-compact-webgpu-int8/resolve/main/";
const MODEL_MB = 313;
const FRAME_LIMIT = 16;
const PAUSE_MS = 600; // paint once typing pauses, not on every keystroke

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/**
 * One painted picture. `label` is what changed since the frame before it.
 * @typedef {{ key: number, prompt: string, label: string, url: string, ms: number | null }} Frame
 */

/**
 * @typedef {{ kind: "loading", progress: { loaded: number, total: number } | null }
 *   | { kind: "ready" }
 *   | { kind: "unsupported" }
 *   | { kind: "error", message: string }} Status
 */

/** @type {Frame[]} */
const opening = OPENING.map((o, i) => ({
  key: i,
  prompt: o.prompt,
  label: labelFor(OPENING[i - 1]?.prompt ?? "", o.prompt),
  url: `./samples/${o.id}.webp`,
  ms: null,
}));

/** @type {Status} */
let status = { kind: "loading", progress: null };
let prompt = opening.at(-1).prompt;
let frames = opening;
let current = opening.at(-1);
let behind = null; // previous picture's URL, shown under the fade-in
let painting = false;
let aboutOpen = false;

let painter = null;
let nextKey = opening.length;
let seed = randomSeed();
let wanted = null; // the one job waiting to be painted: { prompt, seed }
let debounce;
let lastAsked = ""; // seed + prompt of the latest request, to skip repaints of the same thing
let draining = Promise.resolve();

const isReady = () => status.kind === "ready";

// ---------------------------------------------------------------------------
// Elements
// ---------------------------------------------------------------------------

const $ = (id) => document.getElementById(id);

const el = {
  aboutWrap: $("about-wrap"),
  aboutButton: $("about-button"),
  about: $("about"),
  aboutImage: $("about-image"),
  aboutTime: $("about-time"),
  sentence: $("sentence"),
  mirror: $("mirror"),
  input: $("prompt"),
  film: $("film"),
  suggestions: $("suggestions"),
  hero: $("hero"),
};

/** Small helper: create an element with a class and optional text. */
function make(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

// Pieces of the hero picture that persist between renders.
const hero = {
  behind: make("img", "behind"),
  front: null, // replaced whenever the current picture changes, so it fades in again
  frontKey: -1,
  loading: make("div", "overlay"),
  loadingBadge: make("span", "badge"),
  loadingBar: make("div"),
  problem: make("div", "overlay"),
  problemBadge: make("span", "badge wide"),
  time: make("span", "badge time"),
};
hero.behind.alt = "";
const bar = make("div", "bar");
bar.append(hero.loadingBar);
hero.loading.append(hero.loadingBadge, bar);
hero.problem.append(hero.problemBadge);
el.hero.append(hero.loading, hero.problem, hero.time);

// Filmstrip buttons by frame key, so existing thumbnails aren't rebuilt (and don't re-animate).
const filmButtons = new Map();

// Suggestion buttons, built once.
const pickButtons = [];
for (const s of SUGGESTIONS) {
  pickButtons.push(pickButton(s.emoji, s.label, () => useSuggestion(s.prompt)));
}
pickButtons.push(pickButton("🎲", "surprise me", surprise));
el.suggestions.append(...pickButtons);

function pickButton(emoji, label, onClick) {
  const button = make("button", "pill pick");
  const icon = make("span", "", emoji);
  icon.setAttribute("aria-hidden", "true");
  button.append(icon, " " + label);
  button.addEventListener("click", onClick);
  return button;
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

function render() {
  renderSentence();
  renderFilm();
  renderHero();
  renderSuggestions();
  renderAbout();
}

/** The sentence: words already in the picture are ink; words typed since the last paint are blue. */
function renderSentence() {
  if (el.input.value !== prompt) el.input.value = prompt;

  const size =
    prompt.length < 60 ? "xl" : prompt.length < 110 ? "lg" : prompt.length < 180 ? "md" : "sm";
  el.sentence.className = `sentence ${size}`;

  if (prompt) {
    const painted = sharedWords(current.prompt, prompt);
    const pending = prompt.slice(painted.length);
    // Upright <em>, so glyph widths match the textarea and the caret doesn't drift.
    // The zero-width space keeps a trailing newline's height.
    el.mirror.replaceChildren(make("span", "done", painted), make("em", "", pending), "​");
  } else {
    const text = isReady() ? "Describe anything…" : "Getting ready…";
    el.mirror.replaceChildren(make("span", "placeholder", text));
  }
}

function renderFilm() {
  const keys = new Set(frames.map((f) => f.key));
  for (const [key, button] of filmButtons) {
    if (!keys.has(key)) {
      button.remove();
      filmButtons.delete(key);
    }
  }
  // Frames only ever leave from the front and arrive at the back, so appending keeps the order.
  for (const frame of frames) {
    let button = filmButtons.get(frame.key);
    if (!button) {
      button = make("button", "frame");
      button.title = frame.prompt;
      const img = make("img");
      img.src = frame.url;
      img.alt = frame.prompt;
      button.append(img, make("span", "", frame.label));
      button.addEventListener("click", () => open(frame));
      filmButtons.set(frame.key, button);
      el.film.append(button);
    }
    button.classList.toggle("on", frame === current);
  }
}

function renderHero() {
  el.hero.classList.toggle("busy", painting);

  if (behind) {
    hero.behind.src = behind;
    if (!hero.behind.isConnected) el.hero.prepend(hero.behind);
  } else {
    hero.behind.remove();
  }

  if (hero.frontKey !== current.key) {
    const img = make("img");
    img.src = current.url;
    img.alt = current.prompt;
    if (hero.front) hero.front.replaceWith(img);
    else hero.behind.isConnected ? hero.behind.after(img) : el.hero.prepend(img);
    hero.front = img;
    hero.frontKey = current.key;
  }

  hero.loading.hidden = status.kind !== "loading";
  hero.problem.hidden = status.kind !== "unsupported" && status.kind !== "error";
  hero.time.hidden = status.kind !== "ready" || current.ms === null;

  if (status.kind === "loading") {
    const { loadedMB, totalMB } = megabytes();
    hero.loadingBadge.textContent = status.progress
      ? `Getting ready · ${Math.min(loadedMB, totalMB)} / ${totalMB} MB`
      : "Getting ready";
    hero.loadingBar.style.width = `${Math.min(100, (100 * loadedMB) / totalMB)}%`;
  } else if (status.kind === "unsupported") {
    hero.problemBadge.textContent = "This needs WebGPU. Try Chrome or Edge on a laptop or desktop.";
  } else if (status.kind === "error") {
    hero.problemBadge.textContent = status.message;
  } else if (current.ms !== null) {
    hero.time.textContent = seconds(current.ms);
  }
}

function renderSuggestions() {
  for (const button of pickButtons) button.disabled = !isReady();
}

function renderAbout() {
  el.about.hidden = !aboutOpen;
  el.aboutButton.setAttribute("aria-expanded", String(aboutOpen));
  if (aboutOpen) {
    el.aboutImage.src = current.url;
    el.aboutTime.textContent = current.ms !== null ? seconds(current.ms) : "—";
  }
}

function megabytes() {
  const progress = status.kind === "loading" ? status.progress : null;
  return {
    loadedMB: progress ? Math.round(progress.loaded / 1e6) : 0,
    totalMB: progress?.total ? Math.round(progress.total / 1e6) : MODEL_MB,
  };
}

// ---------------------------------------------------------------------------
// Startup
// ---------------------------------------------------------------------------

function start() {
  render();
  if (!hasWebGPU()) {
    status = { kind: "unsupported" };
    render();
    return;
  }
  Painter.load(MODEL_BASE, (progress) => {
    status = { kind: "loading", progress };
    renderHero();
  })
    .then((p) => {
      painter = p;
      status = { kind: "ready" };
      render();
      request(); // repaint the opening prompt on this device
    })
    .catch((err) => {
      console.error(err);
      status = { kind: "error", message: describeError(err) };
      render();
    });
}

function describeError(err) {
  const text = err instanceof Error ? err.message : String(err);
  if (/webgpu|adapter|device|gpu/i.test(text))
    return "Your GPU couldn’t start WebGPU. Try Chrome or Edge on a laptop.";
  if (/quota|memory|allocation/i.test(text))
    return "Not enough memory to load the model on this device.";
  return "Something went wrong loading the model. Reload to try again.";
}

// ---------------------------------------------------------------------------
// Painting
// ---------------------------------------------------------------------------

function randomSeed() {
  return Math.floor(Math.random() * 2 ** 31);
}

/** The new words in `next` when it extends `prev`, otherwise the whole prompt. */
function labelFor(prev, next) {
  if (prev && next.startsWith(prev)) return next.slice(prev.length).replace(/^[\s,.;]+/, "") || next;
  return next;
}

/** Longest shared beginning of two prompts, cut back to a whole word. */
function sharedWords(a, b) {
  let i = 0;
  while (i < a.length && i < b.length && a[i] === b[i]) i++;
  if (i === b.length || i === a.length) return b.slice(0, i);
  return b.slice(0, b.lastIndexOf(" ", i) + 1);
}

/** Paint the text with the current seed, latest request wins. */
function request() {
  const text = prompt.trim();
  if (!text || !isReady()) return;
  const key = `${seed}:${text}`;
  if (key === lastAsked) return;
  lastAsked = key;
  wanted = { prompt: text, seed };
  if (!painting) draining = drain();
}

/** Paint jobs one at a time until nothing new is wanted. */
async function drain() {
  painting = true;
  renderHero();
  while (wanted && painter) {
    const job = wanted;
    wanted = null;
    try {
      const result = await painter.paint(job.prompt, job.seed);
      show(job.prompt, result.url, result.ms);
    } catch (err) {
      console.error(err);
      lastAsked = ""; // let the same prompt be retried
      status = { kind: "error", message: describeError(err) };
      render();
    }
  }
  painting = false;
  renderHero();
}

/** Show a new painting and add it to the filmstrip (replacing a repaint of the same prompt). */
function show(text, url, ms) {
  const last = frames.at(-1);
  const sameAsLast = last?.prompt === text;
  const frame = {
    key: nextKey++,
    prompt: text,
    label: sameAsLast ? last.label : labelFor(last?.prompt ?? "", text),
    url,
    ms,
  };
  const kept = sameAsLast ? frames.slice(0, -1) : frames;
  const next = [...kept, frame].slice(-FRAME_LIMIT);
  const dropped = frames.filter((f) => !next.includes(f));
  const stale = behind;
  behind = current.url;
  current = frame;
  frames = next;

  // Free blob: URLs that no longer appear anywhere on the page.
  const inUse = (u) => u === behind || u === current.url || frames.some((f) => f.url === u);
  for (const u of [stale, ...dropped.map((f) => f.url)]) {
    if (u?.startsWith("blob:") && !inUse(u)) URL.revokeObjectURL(u);
  }

  render();
  el.film.scrollTo({ left: el.film.scrollWidth, behavior: "smooth" });
}

/** Enter: paint what's typed now, then start a fresh sentence with a new seed. */
async function startOver() {
  if (!isReady()) return;
  clearTimeout(debounce);
  request();
  await draining;
  seed = randomSeed();
  prompt = "";
  renderSentence();
  el.input.focus();
}

function useSuggestion(text) {
  prompt = text;
  seed = randomSeed();
  renderSentence();
  request();
}

function surprise() {
  const options = SURPRISES.filter((s) => s !== prompt);
  useSuggestion(options[Math.floor(Math.random() * options.length)]);
}

/** Bring an earlier frame back to the stage. */
function open(frame) {
  if (frame === current) return;
  behind = current.url;
  current = frame;
  prompt = frame.prompt;
  render();
}

function seconds(ms) {
  return `${(ms / 1000).toFixed(1)} s`;
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

el.input.addEventListener("input", () => {
  prompt = el.input.value;
  renderSentence();
  clearTimeout(debounce);
  debounce = setTimeout(request, PAUSE_MS);
});

el.input.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" || event.shiftKey || event.isComposing) return;
  event.preventDefault();
  startOver();
});

el.aboutButton.addEventListener("click", () => {
  aboutOpen = !aboutOpen;
  renderAbout();
});

window.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && aboutOpen) {
    aboutOpen = false;
    renderAbout();
  }
});

window.addEventListener("pointerdown", (event) => {
  if (aboutOpen && !el.aboutWrap.contains(event.target)) {
    aboutOpen = false;
    renderAbout();
  }
});

start();
