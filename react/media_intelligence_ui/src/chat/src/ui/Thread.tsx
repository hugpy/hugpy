/*
 * Thread.tsx — scrollable transcript area.
 *
 * Pared down: this is JUST the message list and its scroll behavior
 * when there ARE chats. The empty-state UI (headline + centered
 * composer) is handled by main.tsx, because convo positions the
 * composer differently in the empty vs loaded states.
 *
 * Sticky-scroll preserved: if you're near the bottom, new turns
 * scroll into view; if you've scrolled up to read, we leave you be.
 */

import { useEffect, useRef, useState } from "react";
import type { ChatEntry } from "./../imports";
import MessageTurn from "./MessageTurn";
import ToolRunTurn from "./ToolRunTurn";

interface ThreadProps {
  chats: ChatEntry[];
  loading: boolean;
  modelLabel: string;
  models?: { key: string; label: string }[];
  onEditQuery?: (chatId: string, newText: string) => void;
  onRegenerate?: (chatId: string, modelKey?: string) => void;
  onDelete?: (chatId: string) => void;
  onOpenFile?: (id: string) => void;
}

export default function Thread({
  chats,
  loading,
  modelLabel,
  models,
  onEditQuery,
  onRegenerate,
  onDelete,
  onOpenFile,
}: ThreadProps): JSX.Element {
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const prevCountRef = useRef(chats.length);
  // Space reserved directly BELOW the newest turn so a short, still-generating
  // turn can actually reach the top of the scroll region (see the effect).
  const [tailSpace, setTailSpace] = useState(0);

  // When a new turn is added — i.e. the user just submitted a prompt — bring the
  // SENT QUERY to the top of the viewport so it leads and its reply streams in
  // below it, instead of docking down by the composer. Keyed off the turn COUNT,
  // not content, so streaming tokens don't re-yank the view. A short new turn
  // has too little beneath it to scroll to the top, so we first reserve ~a
  // viewport of space below it, then scroll (after the spacer paints).
  useEffect(() => {
    const el = scrollRef.current;
    const prevCount = prevCountRef.current;
    prevCountRef.current = chats.length;
    if (!el || chats.length <= prevCount || chats.length === 0) return;
    const last = chats[chats.length - 1];
    const node = el.querySelector<HTMLElement>(
      `[data-conversation-turn="${last.id}"]`,
    );
    if (!node) return;
    setTailSpace(Math.max(0, el.clientHeight - node.offsetHeight));
    requestAnimationFrame(() =>
      requestAnimationFrame(() =>
        node.scrollIntoView({ block: "start", behavior: "smooth" }),
      ),
    );
  }, [chats]);

  return (
    <div ref={scrollRef} className="flex-1 overflow-y-auto">
      <div id="thread" className="group/thread flex flex-col min-h-full">
        <div
          className="
            mx-auto w-full
            px-[var(--thread-content-margin-xs)]
            sm:px-[var(--thread-content-margin-sm)]
            lg:px-[var(--thread-content-margin-lg)]
          "
        >
          <div
            className="
              mx-auto flex-1 pt-2 pb-4
              max-w-[var(--thread-content-max-width-sm)]
              lg:max-w-[var(--thread-content-max-width-lg)]
            "
          >
            {chats.map((chat) =>
              chat.kind === "tool" ? (
                <ToolRunTurn key={chat.id} chat={chat} />
              ) : (
                <MessageTurn
                  key={chat.id}
                  chat={chat}
                  modelLabel={modelLabel}
                  models={models}
                  busy={loading}
                  onEditQuery={onEditQuery}
                  onRegenerate={onRegenerate}
                  onDelete={onDelete}
                  onOpenFile={onOpenFile}
                />
              ),
            )}
            <div aria-hidden="true" style={{ height: tailSpace }} />
          </div>
        </div>
      </div>
    </div>
  );
}
