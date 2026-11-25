# Imagine: flat build

A dependency-free, build-free copy of the app: plain HTML, CSS and ES modules that any static host can serve as-is, including GitHub Pages (from a project subpath like `/imagine/`).

| Path | What it is |
| --- | --- |
| `index.html` | Page markup |
| `style.css` | All styles |
| `js/app.js` | UI state, rendering, debounced latest-wins paint queue, filmstrip |
| `js/engine.js` | Model download + Cache Storage, ONNX Runtime WebGPU session, seeded noise, WebP output |
| `js/tokenizer.js` | T5 Unigram (SentencePiece-compatible) tokenizer |
| `js/prompts.js` | Opening frames, suggestions, "surprise me" prompts |
| `ort/` | Vendored onnxruntime-web 1.22.0 (MIT): `ort.webgpu.min.mjs` plus its JSEP/WASM runtime |
| `samples/` | Pre-painted opening frames shown while the model downloads |

The ~313 MB model is fetched from Hugging Face at runtime (change `MODEL_BASE` in `js/app.js` to host it elsewhere). No COOP/COEP headers are needed: ONNX Runtime runs single-threaded.

## Run locally

Double-click `run.command` (macOS) or `run.bat` (Windows), or use any static server, for example `python3 -m http.server -d flat 8080`, then open <http://localhost:8080>. Opening `index.html` straight from disk (`file://`) won't work, because browsers block ES modules there.

## GitHub Pages

`.github/workflows/pages.yml` publishes this folder on every push to `main`. In the repo, go to **Settings → Pages** and set **Source** to **GitHub Actions**.
