/** Quick-start buttons under the sentence. */
export const SUGGESTIONS = [
  {
    label: "fox under a mushroom",
    emoji: "🦊",
    prompt:
      "A tiny red fox reading a book under a mushroom, soft morning light, storybook illustration",
  },
  {
    label: "cabin + northern lights",
    emoji: "🌌",
    prompt:
      "A cozy cabin in a snowy forest at night, warm windows, northern lights above",
  },
  {
    label: "ramen",
    emoji: "🍜",
    prompt:
      "A bowl of ramen with a soft egg and steam rising, top-down food photography",
  },
];

/** Prompts picked at random by "surprise me". */
export const SURPRISES = [
  "A lighthouse on a cliff during a thunderstorm, dramatic waves, oil painting",
  "A glass teapot filled with a miniature galaxy, studio product photo",
  "An origami crane made of gold leaf on black velvet, macro photography",
  "A koi pond seen from above, autumn leaves on the water, watercolor",
  "A vintage red bicycle leaning against a blue wall in Lisbon, film photo",
  "A floating island with a waterfall and a small castle, fantasy concept art",
  "A robot gardener watering sunflowers, pastel colors, children's book illustration",
  "A hot air balloon over lavender fields at sunset, impressionist painting",
  "A sleepy cat curled up on a stack of old books, warm lamplight, oil painting",
  "A neon-lit alley in Tokyo in the rain, reflections on wet pavement, cinematic",
  "A desert at dusk with giant pastel moons on the horizon, surreal digital art",
  "A single strawberry splashing into milk, high-speed photography",
  "An underwater temple overgrown with coral, rays of sunlight, fantasy art",
  "A whale carrying a tiny library on its back, deep ocean, candlelight",
  "A low-poly geometric lion head, gold facets on a black background, vector art",
  "A city of glowing jellyfish floating above dark ocean skyscrapers, bioluminescent",
];

/**
 * Real outputs from this model (seed 100), one per pause while typing the
 * opening prompt. They fill the filmstrip before the model loads on your device.
 */
export const OPENING = [
  { id: "corgi-0", prompt: "A corgi" },
  { id: "corgi-1", prompt: "A corgi astronaut" },
  { id: "corgi-2", prompt: "A corgi astronaut floating" },
  { id: "corgi-3", prompt: "A corgi astronaut floating in space" },
  { id: "corgi-4", prompt: "A corgi astronaut floating in space, soft studio light" },
];
