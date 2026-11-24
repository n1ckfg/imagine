import * as ort from "onnxruntime-web/webgpu";
import { T5Tokenizer } from "./tokenizer";

const IMAGE_SIZE = 512;
const PROMPT_LENGTH = 256;
const NOISE_SCALE = 2; // the scheduler's init_noise_sigma
const CACHE_NAME = "minit2i-onnx-v2"; // v1 held the 617 MB fp16 model

export type LoadProgress = { loaded: number; total: number };

export type Painting = { url: string; ms: number };

export class Painter {
  private constructor(
    private readonly session: ort.InferenceSession,
    private readonly tokenizer: T5Tokenizer,
  ) {}

  static async load(
    modelBase: string,
    onProgress: (p: LoadProgress) => void,
  ): Promise<Painter> {
    ort.env.wasm.wasmPaths = new URL("ort/", document.baseURI).href;
    ort.env.wasm.numThreads = 1;
    ort.env.logLevel = "error";

    const files = ["minit2i.onnx", "minit2i.onnx.data", "vocab.json"];
    const sizes = await Promise.all(
      files.map((f) => cachedSize(modelBase + f)),
    );
    const total = sizes.reduce((a, b) => a + b, 0);
    const loaded = files.map(() => 0);
    const report = () =>
      onProgress({ loaded: loaded.reduce((a, b) => a + b, 0), total });

    const [graph, weights, vocab] = await Promise.all(
      files.map((f, i) =>
        fetchCached(modelBase + f, (n) => {
          loaded[i] = n;
          report();
        }),
      ),
    );
    const tokenizer = new T5Tokenizer(
      JSON.parse(new TextDecoder().decode(vocab)),
    );
    const session = await ort.InferenceSession.create(graph, {
      executionProviders: ["webgpu"],
      externalData: [{ path: "minit2i.onnx.data", data: weights }],
    });
    return new Painter(session, tokenizer);
  }

  async paint(prompt: string, seed: number): Promise<Painting> {
    const start = performance.now();
    const { ids, mask } = this.tokenizer.encodePadded(prompt, PROMPT_LENGTH);
    const feeds = {
      input_ids: new ort.Tensor("int64", ids, [1, PROMPT_LENGTH]),
      attention_mask: new ort.Tensor("int64", mask, [1, PROMPT_LENGTH]),
      noise: new ort.Tensor("float32", gaussianNoise(seed), [
        1,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
      ]),
    };
    const { image } = await this.session.run(feeds);
    const pixels = image.data as Float32Array;
    image.dispose();
    const ms = performance.now() - start;
    return { url: await toObjectUrl(pixels), ms };
  }
}

export function hasWebGPU(): boolean {
  return typeof navigator !== "undefined" && "gpu" in navigator;
}

async function cachedSize(url: string): Promise<number> {
  const hit = await openCache().then((c) => c?.match(url));
  if (hit) return Number(hit.headers.get("content-length") ?? 0);
  const head = await fetch(url, { method: "HEAD" });
  if (!head.ok) throw new Error(`Could not reach ${url} (${head.status})`);
  return Number(head.headers.get("content-length") ?? 0);
}

async function fetchCached(
  url: string,
  onBytes: (n: number) => void,
): Promise<Uint8Array<ArrayBuffer>> {
  const cache = await openCache();
  const hit = await cache?.match(url);
  if (hit) {
    const bytes = new Uint8Array(await hit.arrayBuffer());
    onBytes(bytes.length);
    return bytes;
  }
  const res = await fetch(url);
  if (!res.ok || !res.body)
    throw new Error(`Download failed for ${url} (${res.status})`);
  const expected = Number(res.headers.get("content-length") ?? 0);
  const bytes = await readAll(res.body, expected, onBytes);
  try {
    await cache?.put(
      url,
      new Response(bytes as Uint8Array<ArrayBuffer>, {
        headers: { "content-length": String(bytes.length) },
      }),
    );
  } catch (err) {
    console.warn(
      "Could not cache model file; it will download again next visit.",
      err,
    );
  }
  return bytes;
}

async function readAll(
  body: ReadableStream<Uint8Array>,
  expected: number,
  onBytes: (n: number) => void,
): Promise<Uint8Array<ArrayBuffer>> {
  const reader = body.getReader();
  let buffer = new Uint8Array(expected || 1 << 20);
  let length = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (length + value.length > buffer.length) {
      const grown = new Uint8Array(
        Math.max(buffer.length * 2, length + value.length),
      );
      grown.set(buffer.subarray(0, length));
      buffer = grown;
    }
    buffer.set(value, length);
    length += value.length;
    onBytes(length);
  }
  return length === buffer.length ? buffer : buffer.slice(0, length);
}

async function openCache(): Promise<Cache | undefined> {
  try {
    // Free space taken by older model versions before storing this one.
    for (const name of await caches.keys()) {
      if (name.startsWith("minit2i-onnx-") && name !== CACHE_NAME) await caches.delete(name);
    }
    return await caches.open(CACHE_NAME);
  } catch {
    return undefined; // private windows and some browsers block Cache Storage
  }
}

function gaussianNoise(seed: number): Float32Array {
  let state = seed >>> 0;
  const random = () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const out = new Float32Array(3 * IMAGE_SIZE * IMAGE_SIZE);
  for (let i = 0; i < out.length; i += 2) {
    const radius = NOISE_SCALE * Math.sqrt(-2 * Math.log(1 - random()));
    const angle = 2 * Math.PI * random();
    out[i] = radius * Math.cos(angle);
    out[i + 1] = radius * Math.sin(angle);
  }
  return out;
}

async function toObjectUrl(chw: Float32Array): Promise<string> {
  const plane = IMAGE_SIZE * IMAGE_SIZE;
  const canvas = new OffscreenCanvas(IMAGE_SIZE, IMAGE_SIZE);
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("2D canvas is unavailable");
  const img = ctx.createImageData(IMAGE_SIZE, IMAGE_SIZE);
  for (let p = 0; p < plane; p++) {
    for (let c = 0; c < 3; c++) {
      const v = Math.max(-1, Math.min(1, chw[c * plane + p])) * 127.5 + 128;
      img.data[p * 4 + c] = Math.min(255, Math.floor(v));
    }
    img.data[p * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  const blob = await canvas.convertToBlob({
    type: "image/webp",
    quality: 0.92,
  });
  return URL.createObjectURL(blob);
}
