/*
 * thinking.ts — split reasoning-model output into <think> content vs answer.
 *
 * Reasoning models (DeepSeek-R1, Qwen3, distills) emit their chain of thought
 * inside <think>…</think> (some variants use <thinking>). The parser is
 * stream-safe: it is re-run on every token append, so it must cope with
 *   - an opening tag whose close hasn't arrived yet (reasoning still live),
 *   - a close tag with no opening tag (llama.cpp chat templates often
 *     pre-open <think> inside the prompt, so the stream starts mid-reasoning),
 *   - a partially-received tag at the end of the buffer (would flash as
 *     literal "<thi" for one token if not trimmed).
 */

const OPEN_TAG = /<think(?:ing)?>/i;
const CLOSE_TAG = /<\/think(?:ing)?>/i;

export interface ParsedThinking {
  /** Concatenated content of all think blocks ("" if none). */
  reasoning: string;
  /** Everything outside think blocks — the user-facing answer. */
  answer: string;
  /** True while an opened think block has not been closed yet. */
  active: boolean;
}

// If the buffer ends mid-tag (e.g. "…answer text<thi"), hide the fragment so
// it never flashes as literal text between tokens.
function trimPartialTag(text: string): string {
  const lt = text.lastIndexOf("<");
  if (lt === -1) return text;
  const tail = text.slice(lt).toLowerCase();
  if ("<think>".startsWith(tail) || "</think>".startsWith(tail)) {
    return text.slice(0, lt);
  }
  if ("<thinking>".startsWith(tail) || "</thinking>".startsWith(tail)) {
    return text.slice(0, lt);
  }
  return text;
}

export function parseThinking(text: string): ParsedThinking {
  let reasoning = "";
  let answer = "";
  let active = false;
  let rest = text;

  // Interleaved thinking yields several blocks; keep them readable when joined.
  const addReasoning = (chunk: string) => {
    reasoning += reasoning && chunk ? `\n\n${chunk}` : chunk;
  };

  // Orphan close tag (template pre-opened the think block in the prompt):
  // everything before the first </think> is reasoning.
  const firstOpen = rest.search(OPEN_TAG);
  const firstClose = rest.search(CLOSE_TAG);
  if (firstClose !== -1 && (firstOpen === -1 || firstClose < firstOpen)) {
    addReasoning(rest.slice(0, firstClose));
    rest = rest.slice(firstClose).replace(CLOSE_TAG, "");
  }

  while (rest) {
    const open = rest.match(OPEN_TAG);
    if (!open || open.index === undefined) {
      answer += rest;
      break;
    }
    answer += rest.slice(0, open.index);
    rest = rest.slice(open.index + open[0].length);

    const close = rest.match(CLOSE_TAG);
    if (!close || close.index === undefined) {
      addReasoning(rest);
      active = true;
      break;
    }
    addReasoning(rest.slice(0, close.index));
    rest = rest.slice(close.index + close[0].length);
  }

  return {
    reasoning: (active ? trimPartialTag(reasoning) : reasoning).trim(),
    answer: trimPartialTag(answer).replace(/^\s+/, ""),
    active,
  };
}
