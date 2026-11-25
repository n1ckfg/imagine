// T5 Unigram tokenizer (SentencePiece-compatible Viterbi), matches the
// model's tokenizer.json on every prompt we tested, including accents and emoji.

const PAD_ID = 0;
const EOS_ID = 1;
const UNK_ID = 2;
const FIRST_PIECE_ID = 3;

export class T5Tokenizer {
  /**
   * @param {Array<[piece: string, score: number]>} vocab  the model's vocab.json
   */
  constructor(vocab) {
    /** piece -> [id, score] */
    this.pieces = new Map();
    let maxLength = 0;
    let minScore = 0;
    vocab.forEach(([piece, score], id) => {
      if (id < FIRST_PIECE_ID) return;
      this.pieces.set(piece, [id, score]);
      maxLength = Math.max(maxLength, piece.length);
      minScore = Math.min(minScore, score);
    });
    this.maxPieceLength = maxLength;
    this.unkScore = minScore - 10;
  }

  /** Token ids for `text`, ending with EOS. */
  encode(text) {
    const words = text.normalize("NFKC").split(/\s+/).filter(Boolean);
    const ids = words.flatMap((word) => this.encodeWord("▁" + word));
    ids.push(EOS_ID);
    return ids;
  }

  /**
   * Returns input ids and attention mask padded to `length`, truncating overlong prompts.
   * @returns {{ ids: BigInt64Array, mask: BigInt64Array }}
   */
  encodePadded(text, length) {
    let tokens = this.encode(text);
    if (tokens.length > length) tokens = [...tokens.slice(0, length - 1), EOS_ID];
    const ids = new BigInt64Array(length).fill(BigInt(PAD_ID));
    const mask = new BigInt64Array(length);
    tokens.forEach((token, i) => {
      ids[i] = BigInt(token);
      mask[i] = 1n;
    });
    return { ids, mask };
  }

  /** Best-scoring split of one word into vocabulary pieces (Viterbi). */
  encodeWord(word) {
    const n = word.length;
    const best = new Array(n + 1).fill(-Infinity);
    const back = new Array(n + 1); // back[j] = [start, id] of the piece ending at j
    best[0] = 0;
    for (let i = 0; i < n; i++) {
      if (best[i] === -Infinity) continue;
      let hasSingleChar = false;
      for (let len = 1; len <= this.maxPieceLength && i + len <= n; len++) {
        const entry = this.pieces.get(word.slice(i, i + len));
        if (!entry) continue;
        if (len === 1) hasSingleChar = true;
        const score = best[i] + entry[1];
        if (score > best[i + len]) {
          best[i + len] = score;
          back[i + len] = [i, entry[0]];
        }
      }
      if (!hasSingleChar && best[i] + this.unkScore > best[i + 1]) {
        best[i + 1] = best[i] + this.unkScore;
        back[i + 1] = [i, UNK_ID];
      }
    }
    const out = [];
    for (let j = n; j > 0; j = back[j][0]) out.push(back[j][1]);
    out.reverse();
    // SentencePiece merges consecutive unknown pieces into one.
    return out.filter((id, k) => !(id === UNK_ID && out[k - 1] === UNK_ID));
  }
}
