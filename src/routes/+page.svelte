<script lang="ts">
  import { onMount, tick } from "svelte";
  import { Painter, hasWebGPU, type LoadProgress } from "../lib/engine";
  import { OPENING, SUGGESTIONS, SURPRISES } from "../lib/prompts";

  const MODEL_BASE =
    import.meta.env.VITE_MODEL_BASE ??
    "https://huggingface.co/cfahlgren1/MiniT2I-B-16-MF-compact-webgpu-int8/resolve/main/";
  const MODEL_MB = 313;
  const FRAME_LIMIT = 16;
  const PAUSE_MS = 600; // paint once typing pauses, not on every keystroke
  const LINKS = {
    paper: "https://huggingface.co/MiniT2I/MiniT2I-B-16-MF-jax",
    compact: "https://huggingface.co/lox9973/MiniT2I-B-16-MF-compact",
    webgpu: "https://huggingface.co/cfahlgren1/MiniT2I-B-16-MF-compact-webgpu-int8",
  };

  /** One painted picture. `label` is what changed since the frame before it. */
  type Frame = {
    key: number;
    prompt: string;
    label: string;
    url: string;
    ms: number | null;
  };

  type Status =
    | { kind: "loading"; progress: LoadProgress | null }
    | { kind: "ready" }
    | { kind: "unsupported" }
    | { kind: "error"; message: string };

  const opening: Frame[] = OPENING.map((o, i) => ({
    key: i,
    prompt: o.prompt,
    label: labelFor(OPENING[i - 1]?.prompt ?? "", o.prompt),
    url: `./samples/${o.id}.webp`,
    ms: null,
  }));

  let status = $state<Status>({ kind: "loading", progress: null });
  let prompt = $state(opening.at(-1)!.prompt);
  let frames = $state<Frame[]>(opening);
  let current = $state<Frame>(opening.at(-1)!);
  let behind = $state<string | null>(null); // previous picture, shown under the fade-in
  let painting = $state(false);

  let painter: Painter | null = null;
  let nextKey = opening.length;
  let seed = randomSeed();
  let wanted: { prompt: string; seed: number } | null = null;
  let debounce: ReturnType<typeof setTimeout> | undefined;
  let lastAsked = ""; // seed + prompt of the latest request, to skip repaints of the same thing
  let draining: Promise<void> = Promise.resolve();
  let input: HTMLTextAreaElement | undefined = $state();
  let film: HTMLElement | undefined = $state();
  let aboutOpen = $state(false);
  let about: HTMLElement | undefined = $state();

  const ready = $derived(status.kind === "ready");
  const loadedMB = $derived(
    status.kind === "loading" && status.progress ? Math.round(status.progress.loaded / 1e6) : 0,
  );
  const totalMB = $derived(
    status.kind === "loading" && status.progress?.total
      ? Math.round(status.progress.total / 1e6)
      : MODEL_MB,
  );
  const percent = $derived(Math.min(100, (100 * loadedMB) / totalMB));
  // Words already in the picture are ink; words typed since the last paint are blue.
  const painted = $derived(sharedWords(current.prompt, prompt));
  const pending = $derived(prompt.slice(painted.length));
  const size = $derived(
    prompt.length < 60 ? "xl" : prompt.length < 110 ? "lg" : prompt.length < 180 ? "md" : "sm",
  );

  onMount(() => {
    if (!hasWebGPU()) {
      status = { kind: "unsupported" };
      return;
    }
    Painter.load(MODEL_BASE, (progress) => {
      status = { kind: "loading", progress };
    })
      .then((p) => {
        painter = p;
        status = { kind: "ready" };
        request(); // repaint the opening prompt on this device
      })
      .catch((err: unknown) => {
        console.error(err);
        status = { kind: "error", message: describeError(err) };
      });
  });

  function describeError(err: unknown): string {
    const text = err instanceof Error ? err.message : String(err);
    if (/webgpu|adapter|device|gpu/i.test(text))
      return "Your GPU couldn’t start WebGPU. Try Chrome or Edge on a laptop.";
    if (/quota|memory|allocation/i.test(text))
      return "Not enough memory to load the model on this device.";
    return "Something went wrong loading the model. Reload to try again.";
  }

  function randomSeed(): number {
    return Math.floor(Math.random() * 2 ** 31);
  }

  /** The new words in `next` when it extends `prev`, otherwise the whole prompt. */
  function labelFor(prev: string, next: string): string {
    if (prev && next.startsWith(prev)) return next.slice(prev.length).replace(/^[\s,.;]+/, "") || next;
    return next;
  }

  /** Longest shared beginning of two prompts, cut back to a whole word. */
  function sharedWords(a: string, b: string): string {
    let i = 0;
    while (i < a.length && i < b.length && a[i] === b[i]) i++;
    if (i === b.length || i === a.length) return b.slice(0, i);
    return b.slice(0, b.lastIndexOf(" ", i) + 1);
  }

  /** Paint the text with the current seed, latest request wins. */
  function request() {
    const text = prompt.trim();
    if (!text || !ready) return;
    const key = `${seed}:${text}`;
    if (key === lastAsked) return;
    lastAsked = key;
    wanted = { prompt: text, seed };
    if (!painting) draining = drain();
  }

  async function drain() {
    painting = true;
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
      }
    }
    painting = false;
  }

  /** Show a new painting and add it to the filmstrip (replacing a repaint of the same prompt). */
  async function show(text: string, url: string, ms: number) {
    const last = frames.at(-1);
    const sameAsLast = last?.prompt === text;
    const frame: Frame = {
      key: nextKey++,
      prompt: text,
      label: sameAsLast ? last.label : labelFor(last?.prompt ?? "", text),
      url,
      ms,
    };
    const kept = sameAsLast ? frames.slice(0, -1) : frames;
    const next = [...kept, frame].slice(-FRAME_LIMIT);
    const dropped = [...frames.filter((f) => !next.includes(f))];
    const stale = behind;
    behind = current.url;
    current = frame;
    frames = next;
    const inUse = (u: string) => u === behind || u === current.url || frames.some((f) => f.url === u);
    for (const u of [stale, ...dropped.map((f) => f.url)]) {
      if (u?.startsWith("blob:") && !inUse(u)) URL.revokeObjectURL(u);
    }
    await tick();
    film?.scrollTo({ left: film.scrollWidth, behavior: "smooth" });
  }

  function onInput() {
    clearTimeout(debounce);
    debounce = setTimeout(request, PAUSE_MS);
  }

  /** Enter: paint what's typed now, then start a fresh sentence with a new seed. */
  async function startOver() {
    if (!ready) return;
    clearTimeout(debounce);
    request();
    await draining;
    seed = randomSeed();
    prompt = "";
    input?.focus();
  }

  function onKeydown(event: KeyboardEvent) {
    if (event.key !== "Enter" || event.shiftKey || event.isComposing) return;
    event.preventDefault();
    void startOver();
  }

  function useSuggestion(text: string) {
    prompt = text;
    seed = randomSeed();
    request();
  }

  function surprise() {
    const options = SURPRISES.filter((s) => s !== prompt);
    useSuggestion(options[Math.floor(Math.random() * options.length)]);
  }

  function open(frame: Frame) {
    if (frame === current) return;
    behind = current.url;
    current = frame;
    prompt = frame.prompt;
  }

  function seconds(ms: number): string {
    return `${(ms / 1000).toFixed(1)} s`;
  }
</script>

<svelte:head>
  <title>Imagine</title>
  <meta
    name="description"
    content="Type anything and watch it paint. A 512 × 512 image in about half a second, made on your own laptop in the browser."
  />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin="anonymous" />
  <link
    href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&display=swap"
    rel="stylesheet"
  />
</svelte:head>

<svelte:window
  onkeydown={(e) => e.key === "Escape" && (aboutOpen = false)}
  onpointerdown={(e) => aboutOpen && !about?.contains(e.target as Node) && (aboutOpen = false)}
/>

<main>
  <header>
    <a class="pill strong" href={LINKS.compact} target="_blank" rel="noreferrer">
      <span aria-hidden="true">🤗</span> Imagine
    </a>
    <div class="about-wrap" bind:this={about}>
      <button
        class="pill icon"
        aria-label="About"
        aria-expanded={aboutOpen}
        aria-controls="about"
        onclick={() => (aboutOpen = !aboutOpen)}
      >
        <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
          <circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="1.8" />
          <path d="M12 11v6M12 7.5v.01" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" />
        </svg>
      </button>
      {#if aboutOpen}
        <div class="about" id="about">
          <img src={current.url} alt="" />
          <dl>
            <div><dt>Picture</dt><dd>512 × 512</dd></div>
            <div><dt>Steps</dt><dd>4</dd></div>
            <div><dt>Time</dt><dd>{current.ms !== null ? seconds(current.ms) : "—"}</dd></div>
            <div><dt>Painted on</dt><dd>your GPU</dd></div>
            <div><dt>Uploaded</dt><dd>nothing</dd></div>
            <div class="total"><dt>Cost</dt><dd>$0.00</dd></div>
          </dl>
          <p class="thanks">Made possible by</p>
          <a href={LINKS.paper} target="_blank" rel="noreferrer">
            <span>MiniT2I</span><small>Wang et al. · Kaiming He</small>
          </a>
          <a href={LINKS.compact} target="_blank" rel="noreferrer">
            <span>Compact model</span><small>lox9973</small>
          </a>
          <a href={LINKS.webgpu} target="_blank" rel="noreferrer">
            <span>WebGPU weights</span><small>ONNX · int8</small>
          </a>
        </div>
      {/if}
    </div>
  </header>

  <section class="stage">
    <div class="words">
      <div class={`sentence ${size}`}>
        <div class="mirror" aria-hidden="true">
          {#if prompt}
            <span class="done">{painted}</span><em>{pending}</em>&#8203;
          {:else}
            <span class="placeholder">{ready ? "Describe anything…" : "Getting ready…"}</span>
          {/if}
        </div>
        <textarea
          bind:this={input}
          bind:value={prompt}
          oninput={onInput}
          onkeydown={onKeydown}
          maxlength="600"
          rows="1"
          aria-label="Describe a picture"
          autocomplete="off"
          spellcheck="false"
        ></textarea>
      </div>

      <div class="film" bind:this={film} aria-label="Every pause, painted">
        {#each frames as f (f.key)}
          <button class="frame" class:on={f === current} onclick={() => open(f)} title={f.prompt}>
            <img src={f.url} alt={f.prompt} />
            <span>{f.label}</span>
          </button>
        {/each}
      </div>

      <p class="hint">Keep typing. Every pause repaints the picture. Press Enter to start a new one.</p>

      <div class="suggestions">
        {#each SUGGESTIONS as s (s.label)}
          <button class="pill pick" onclick={() => useSuggestion(s.prompt)} disabled={!ready}>
            <span aria-hidden="true">{s.emoji}</span>
            {s.label}
          </button>
        {/each}
        <button class="pill pick" onclick={surprise} disabled={!ready}>
          <span aria-hidden="true">🎲</span> surprise me
        </button>
      </div>
    </div>

    <figure class="hero" class:busy={painting}>
      {#if behind}
        <img class="behind" src={behind} alt="" />
      {/if}
      {#key current.key}
        <img src={current.url} alt={current.prompt} />
      {/key}
      {#if status.kind === "loading"}
        <div class="overlay">
          <span class="badge">
            {status.progress ? `Getting ready · ${Math.min(loadedMB, totalMB)} / ${totalMB} MB` : "Getting ready"}
          </span>
          <div class="bar"><div style:width={`${percent}%`}></div></div>
        </div>
      {:else if status.kind === "unsupported" || status.kind === "error"}
        <div class="overlay">
          <span class="badge wide">
            {status.kind === "unsupported"
              ? "This needs WebGPU. Try Chrome or Edge on a laptop or desktop."
              : status.message}
          </span>
        </div>
      {:else if current.ms !== null}
        <span class="badge time">{seconds(current.ms)}</span>
      {/if}
    </figure>
  </section>
</main>

<style>
  :global(html),
  :global(body) {
    margin: 0;
    background: #f5f5f7;
    color: #1d1d1f;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
  }

  main {
    --hero: min(640px, calc(100dvh - 160px), 46vw);
    --serif: "Instrument Serif", Georgia, "Times New Roman", serif;
    --blue: #0a84ff;
    min-height: 100dvh;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    padding: 28px 32px 32px;
  }

  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
  }

  .pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    min-height: 40px;
    padding: 0 18px;
    box-sizing: border-box;
    border: 0;
    border-radius: 999px;
    background: #fff;
    color: #6e6e73;
    font: inherit;
    font-size: 15px;
    text-decoration: none;
    white-space: nowrap;
    box-shadow: 0 1px 2px rgb(0 0 0 / 0.04), 0 8px 30px rgb(0 0 0 / 0.06);
  }
  .pill.strong {
    color: #1d1d1f;
    font-weight: 600;
  }
  .pill.icon {
    width: 40px;
    padding: 0;
    justify-content: center;
    cursor: pointer;
  }

  /* About: a receipt for the current picture. */
  .about-wrap {
    position: relative;
  }
  .about {
    position: absolute;
    right: 0;
    top: 50px;
    z-index: 5;
    width: min(320px, calc(100vw - 32px));
    max-height: calc(100dvh - 100px);
    overflow-y: auto;
    overscroll-behavior: contain;
    padding: 16px 18px 6px;
    box-sizing: border-box;
    border-radius: 20px;
    background: #fff;
    box-shadow: 0 1px 2px rgb(0 0 0 / 0.04), 0 18px 50px rgb(0 0 0 / 0.14);
    font-size: 14px;
    animation: pop 0.16s ease-out;
  }
  .about > img {
    display: block;
    width: 100%;
    aspect-ratio: 1;
    object-fit: cover;
    border-radius: 12px;
    background: #e8e8ed;
  }
  .about dl {
    margin: 12px 0 0;
    font-variant-numeric: tabular-nums;
  }
  .about dl div {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    padding: 4px 0;
  }
  .about dt {
    color: #6e6e73;
  }
  .about dd {
    margin: 0;
  }
  .about .total {
    margin-top: 6px;
    padding-top: 10px;
    border-top: 1px dashed #d2d2d7;
    font-size: 17px;
    font-weight: 700;
  }
  .about .total dt {
    color: #1d1d1f;
  }
  .about .thanks {
    margin: 14px 0 2px;
    color: #86868b;
    font-size: 12px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }
  .about a {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 12px;
    padding: 10px 0;
    border-top: 1px solid #f0f0f2;
    color: #1d1d1f;
    text-decoration: none;
  }
  .about small {
    min-width: 0;
    color: #86868b;
    font-size: 13px;
    text-align: right;
  }
  @media (hover: hover) {
    .about a:hover span {
      text-decoration: underline;
      text-underline-offset: 3px;
    }
  }

  /* Stage: the sentence on the left, the picture on the right. */
  .stage {
    flex: 1;
    display: grid;
    grid-template-columns: minmax(0, 1fr) var(--hero);
    align-items: center;
    gap: clamp(32px, 5vw, 72px);
    padding: 24px clamp(0px, 3vw, 56px) 8px;
  }
  .words {
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 28px;
  }

  .sentence {
    position: relative;
    font-family: var(--serif);
    font-weight: 400;
    line-height: 1.04;
    letter-spacing: -0.01em;
    font-size: var(--size);
  }
  .sentence.xl {
    --size: clamp(44px, 5.4vw, 80px);
  }
  .sentence.lg {
    --size: clamp(38px, 4.4vw, 64px);
  }
  .sentence.md {
    --size: clamp(32px, 3.4vw, 50px);
  }
  .sentence.sm {
    --size: clamp(26px, 2.6vw, 38px);
  }
  .mirror,
  .sentence textarea {
    margin: 0;
    padding: 0;
    font: inherit;
    letter-spacing: inherit;
    white-space: pre-wrap;
    overflow-wrap: break-word;
  }
  .mirror {
    min-height: 1.04em;
  }
  .mirror .done {
    color: #1d1d1f;
  }
  /* Upright, not italic: the mirror must match the textarea's glyph widths or the caret drifts. */
  .mirror em {
    font-style: normal;
    color: var(--blue);
  }
  .mirror .placeholder {
    color: #c7c7cc;
  }
  .sentence textarea {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    border: 0;
    outline: 0;
    resize: none;
    overflow: hidden;
    background: transparent;
    color: transparent;
    caret-color: var(--blue);
  }
  .sentence textarea::selection {
    background: rgb(10 132 255 / 0.2);
  }

  .film {
    display: flex;
    gap: 10px;
    overflow-x: auto;
    overscroll-behavior-x: contain;
    scrollbar-width: none;
    padding: 4px 4px 2px;
    margin: -4px -4px 0;
  }
  .frame {
    flex: none;
    width: 92px;
    padding: 0;
    border: 0;
    background: none;
    font: inherit;
    text-align: left;
    cursor: pointer;
    animation: slide 0.35s cubic-bezier(0.2, 0.9, 0.3, 1.1);
  }
  .frame img {
    display: block;
    width: 92px;
    height: 92px;
    border-radius: 12px;
    object-fit: cover;
    background: #e8e8ed;
    opacity: 0.85;
    transition: opacity 0.15s;
  }
  .frame span {
    display: block;
    margin-top: 6px;
    color: #86868b;
    font-size: 12px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .frame.on img {
    opacity: 1;
    box-shadow: 0 0 0 2px #fff, 0 0 0 4px var(--blue);
  }
  .frame.on span {
    color: #1d1d1f;
  }

  .hint {
    margin: 0;
    color: #86868b;
    font-size: 15px;
  }
  .suggestions {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }
  .pick {
    min-height: 36px;
    padding: 0 14px;
    color: #1d1d1f;
    font-size: 14px;
    cursor: pointer;
  }
  .pick:disabled {
    cursor: default;
    opacity: 0.5;
  }
  @media (hover: hover) {
    .pick:not(:disabled):hover {
      box-shadow: 0 1px 2px rgb(0 0 0 / 0.06), 0 4px 16px rgb(0 0 0 / 0.1);
    }
    .frame:hover img {
      opacity: 1;
    }
  }

  .hero {
    position: relative;
    margin: 0;
    width: var(--hero);
    aspect-ratio: 1;
    border-radius: 8px;
    overflow: hidden;
    background: #e8e8ed;
    box-shadow: 0 30px 90px rgb(0 0 0 / 0.18);
  }
  .hero img {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
    animation: arrive 0.24s ease-out;
  }
  .hero img.behind {
    animation: none;
  }
  .hero.busy img {
    filter: saturate(0.96);
  }

  .overlay {
    position: absolute;
    inset: auto 0 0 0;
    padding: 16px;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 10px;
    background: linear-gradient(to top, rgb(0 0 0 / 0.35), transparent);
  }
  .badge {
    padding: 6px 12px;
    border-radius: 999px;
    background: rgb(255 255 255 / 0.92);
    color: #1d1d1f;
    font-size: 14px;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    backdrop-filter: blur(8px);
  }
  .badge.wide {
    font-weight: 500;
    line-height: 1.35;
    max-width: 100%;
  }
  .badge.time {
    position: absolute;
    left: 16px;
    bottom: 16px;
  }
  .bar {
    width: 100%;
    height: 4px;
    border-radius: 2px;
    background: rgb(255 255 255 / 0.35);
    overflow: hidden;
  }
  .bar div {
    height: 100%;
    background: #fff;
    transition: width 0.2s linear;
  }

  /* Narrow screens: picture first, sentence under it. */
  @media (max-width: 900px) {
    main {
      --hero: min(640px, calc(100vw - 32px));
      padding: 16px 16px 28px;
    }
    .stage {
      grid-template-columns: minmax(0, 1fr);
      justify-items: center;
      align-items: start;
      gap: 24px;
      padding: 16px 0 0;
    }
    .hero {
      order: -1;
    }
    .words {
      width: 100%;
      gap: 20px;
    }
    .sentence.xl {
      --size: clamp(34px, 9vw, 56px);
    }
    .sentence.lg {
      --size: clamp(30px, 7.5vw, 46px);
    }
    .sentence.md {
      --size: clamp(26px, 6vw, 38px);
    }
    .sentence.sm {
      --size: clamp(22px, 5vw, 30px);
    }
    .frame,
    .frame img {
      width: 72px;
    }
    .frame img {
      height: 72px;
    }
  }

  @keyframes pop {
    from {
      opacity: 0;
      transform: translateY(-4px) scale(0.98);
    }
  }
  @keyframes arrive {
    from {
      opacity: 0;
      filter: blur(10px);
      transform: scale(1.03);
    }
  }
  @keyframes slide {
    from {
      opacity: 0;
      transform: translateX(16px);
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .hero img,
    .frame,
    .about {
      animation: none;
    }
  }
</style>
