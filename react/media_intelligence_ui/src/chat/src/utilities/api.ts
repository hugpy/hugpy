// LiveChat/api/chatApi.ts
//
// Streaming client for the DeepCoder chat host.
//
// Two ideas worth internalizing here:
//
// 1. **Discriminated union over a `type` field**. StreamEvent is one of three
//    shapes; TypeScript narrows on `event.type` so each branch only sees the
//    fields that actually exist. This is the TS equivalent of your Pydantic
//    schemas — same "no ad-hoc objects" idea, just on the wire.
//
// 2. **Async generators (`async function*`) for streams**. They turn a network
//    stream into a `for await (...)` loop in the consumer. The component
//    doesn't have to know about ReadableStream, decoders, or buffering —
//    it just iterates events. This is the TS analog of your Python iterator.
import {type ChatRequest,type StreamEvent,type GenerateChatResponse} from './../imports';
import {getApiBase} from './functions';
import {hugpyConfig} from '../../../config';

// Tag every chat/prompt request with the media arm's dedicated pool so the
// narrator + router route to reserved workers (same pool as /ml). No-op when
// the pool is empty; the server falls back to LOCAL if no pool worker exists.
function withPool<T extends object>(body: T): T & { pool?: string } {
  return hugpyConfig.pool ? { ...body, pool: hugpyConfig.pool } : body;
}
/**
 * Stream chat events from POST /chat.
 *
 * Usage:
 *   const ctrl = new AbortController();
 *   for await (const evt of streamChat(req, ctrl.signal)) {
 *     if (evt.type === "token") append(evt.text);
 *     else if (evt.type === "done")  finalize(evt.finish_reason);
 *     else if (evt.type === "error") fail(evt.message);
 *   }
 *
 * Cancel by calling `ctrl.abort()`. The fetch rejects with an AbortError,
 * which propagates out of the generator — caller catches it.
 */
function assertValidChatRequest(request: ChatRequest): void {
  const userMessages = request.messages.filter(
    (message) => message.role === "user" && message.content.trim(),
  );

  if (userMessages.length === 0) {
    throw new Error("Refusing to send chat request without a user message.");
  }
}

export async function* streamChat(
  request: ChatRequest,
  signal?: AbortSignal,
): AsyncGenerator<StreamEvent> {
  assertValidChatRequest(request);

  // current hugpy: native SSE chat at /chat/stream (was /deepcoder/chat).
  // Body is unchanged; the SSE event union (token/done/error, + ignorable status) matches.
  const response = await fetch(`${getApiBase()}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(withPool(request)),
    signal,
  });
  if (!response.ok || !response.body) {
    const text = await response.text().catch(() => "");
    throw new Error(`Chat failed: HTTP ${response.status} ${text}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE delimits events with a blank line. A network chunk may contain
      // many events or a partial event — buffer until we hit "\n\n".
      let idx: number;
      while ((idx = buffer.indexOf("\n\n")) !== -1) {
        const rawEvent = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);

        for (const line of rawEvent.split("\n")) {
          if (!line.startsWith("data: ")) continue;
          const json = line.slice(6);
          try {
            yield JSON.parse(json) as StreamEvent;
          } catch (err) {
            console.warn("[streamChat] malformed event", json, err);
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
export async function generateChat(
  request: ChatRequest,
  signal?: AbortSignal,
): Promise<GenerateChatResponse> {
  // current hugpy: one /prompt verb dispatches by `task`; text-generation is the chat task.
  const response = await fetch(`${getApiBase()}/prompt`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(withPool({ ...request, task: "text-generation" })),
    signal,
  });

  const text = await response.text();

  let data: GenerateChatResponse;

  try {
    data = JSON.parse(text) as GenerateChatResponse;
  } catch {
    data = {
      ok: response.ok,
      request_id: request.request_id,
      text,
    };
  }

  if (!response.ok || data.ok === false) {
    throw new Error(data.error || `Generate failed: HTTP ${response.status}`);
  }

  return data;
}
/**
 * Tell the server to stop generating tokens for this request_id. Best-effort.
 * The AbortSignal already kills the client side; this stops the GPU work too.
 */
export async function cancelChat(requestId: string): Promise<void> {
  try {
    // current hugpy: cancel by request_id in the PATH (was POST /deepcoder/cancel + body).
    await fetch(`${getApiBase()}/llm/chat/cancel/${requestId}`, {
      method: "POST",
    });
  } catch (err) {
    console.warn("[cancelChat] failed", err);
  }
}
