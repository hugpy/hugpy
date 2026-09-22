/*
 * ChatResponse.tsx — assistant content rendering.
 *
 * Convo renders markdown into structured prose. We don't have a markdown
 * pipeline here, but we approximate the visual: a paragraph stream that
 * wraps naturally, in --text-primary at 15/24 with relaxed spacing.
 *
 * Three branches:
 *   queued/thinking (no text yet)  → typing dots inline with a soft label
 *   array response                 → one <p> per element
 *   string response (default)      → split on blank lines into paragraphs;
 *                                    if the string looks code-heavy (long
 *                                    no-space runs / lots of newlines /
 *                                    obvious code shape) we drop to <pre>.
 */

import type { ChatEntry } from "./../imports";
import { parseThinking } from "./../utilities";
import ThinkingBlock from "./ThinkingBlock";

interface ChatResponseProps {
  chat: ChatEntry;
  modelLabel: string;
}

// crude heuristic: looks code-y if it has lots of newlines + few sentences,
// or contains a fenced block marker. Avoids the urge to <pre>-wrap real prose.
function looksLikeCode(text: string): boolean {
  if (text.includes("```")) return true;
  const newlines = (text.match(/\n/g) ?? []).length;
  const sentences = (text.match(/[.!?](\s|$)/g) ?? []).length;
  return newlines > 6 && sentences < 3;
}

export default function ChatResponse({
  chat,
  modelLabel,
}: ChatResponseProps): JSX.Element {
  const status = chat.status ?? "complete";

  if ((status === "queued" || status === "thinking") && !chat.response) {
    return (
      <div className="flex items-center gap-2 text-[15px] text-token-text-secondary">
        <span className="typing-dots" aria-hidden>
          <span />
          <span />
          <span />
        </span>
        <span>{chat.thinking || `${modelLabel} is thinking…`}</span>
      </div>
    );
  }

  if (Array.isArray(chat.response)) {
    return (
      <div className="flex flex-col gap-3">
        {chat.response.map((line, index) => (
          <p
            key={index}
            className="text-[15px] leading-7 text-token-text-primary whitespace-pre-wrap"
          >
            {line}
          </p>
        ))}
      </div>
    );
  }

  // Reasoning models wrap their chain of thought in <think>…</think>. Split it
  // out so it renders as a collapsible stream instead of leaking into the prose.
  const { reasoning, answer, active } = parseThinking(chat.response ?? "");
  const thinkingBlock =
    reasoning || active ? (
      <ThinkingBlock reasoning={reasoning} active={active} />
    ) : null;

  if (answer && looksLikeCode(answer)) {
    return (
      <div className="flex flex-col gap-3">
        {thinkingBlock}
        <pre
          className="
            m-0 overflow-x-auto rounded-xl px-4 py-3
            text-[13.5px] leading-6 font-mono text-token-text-primary
            whitespace-pre-wrap break-words
          "
          style={{ background: "var(--bg-tertiary)" }}
        >
          {answer}
        </pre>
      </div>
    );
  }

  // Plain prose — split on blank lines into paragraphs.
  const paragraphs = answer ? answer.split(/\n{2,}/) : [];

  return (
    <div className="flex flex-col gap-3">
      {thinkingBlock}
      {paragraphs.map((para, index) => (
        <p
          key={index}
          className="text-[15px] leading-7 text-token-text-primary whitespace-pre-wrap"
        >
          {para}
        </p>
      ))}
    </div>
  );
}
