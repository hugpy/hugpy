/*
 * main.tsx — HugpyChat page shell.
 *
 * Two distinct layouts, switched on `chats.length === 0`:
 *
 *   EMPTY STATE
 *     Centered column: "What are you working on?" headline, composer
 *     directly beneath it. The thread area is empty. This is convo's
 *     blank-canvas pose.
 *
 *   LOADED STATE
 *     Thread scrolls in the middle, composer DOCKS at the bottom in a
 *     sticky shell, disclaimer underneath. Same composer component
 *     either way — only the wrapper differs.
 *
 * The sampling controls (model / tokens / temp / top-p / stream) are
 * always visible below the composer regardless of state.
 *
 * All chat-domain logic (streaming, history building, request lifecycle)
 * is preserved 1:1 from the previous main.tsx.
 */
import './styles/chat.css'
import { useEffect, useMemo, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";

import {
  Sidebar,
  ThreadHeader,
  Thread,
  Composer,
  ComposerControls,
  Disclaimer,
} from "./ui";

// HugpyConsole: the classic tool console, retained as a sidebar-switchable view
// alongside the chat (chat is the default; the sidebar toggles between them).
import { HugpyConsole } from "../../console";

import {
  cancelChat,
  streamChat,
  generateChat,
  createChatId,
  nowIso,
} from "./utilities";

import {
  DEFAULT_MODEL,
  fetchChatModelOptions,
  getChatModelLabel,
  getChatModelOption,
  getChatModelOptions,
  type ChatMessage,
  type ChatModelOption,
  type ChatRequest,
  type ChatEntry,

} from "./imports";

// Chat-first tool integration: the /ml catalog (registered by the side-effect
// import below), the tool runner, the result types, and the inline tray.
import "../../console/src/imports/pages/pagesBuiltin";
import "../../console/src/imports/chain/chainsBuiltin";
import { dispatchTool } from "./utilities/dispatch";
import { runChain, extractText } from "../../console/src/imports/chain/chainRuntime";
import {
  runDocumentIntelligence,
  narrationContextFor,
  failureReasonFor,
  supportsIntelligence,
} from "./utilities/mediaIntelligence";
import type {
  PageSpec,
  MediaInputValue,
} from "../../console/src/imports/pages/pageSpec";
import { okValue, errorOf } from "../../transport/client";
import { getPage, listPages } from "../../console/src/imports/pages/pagesRegistry";
import { routeTool, serializeToolResult } from "./utilities/toolRouting";
import ToolTray from "./ui/ToolTray";
import AttachmentBar from "./ui/AttachmentBar";
import {
  uploadAttachments,
  deleteUploadedFile,
  detectCategory,
  type UploadedFileRef,
  type MediaCategory,
  type StoredFile,
} from "./utilities/fileUpload";
import { loadFiles, saveFiles } from "./utilities/fileStore";
import FilePreview from "./ui/FilePreview";
import {
  loadConversations,
  saveConversations,
  upsertConversation,
  deriveTitle,
  type Conversation,
} from "./utilities/chatHistory";
// Canned-demo seed (null in live/normal mode): opens the chat already populated
// with the worked earnings-call example.
import { demoSeed } from "../../demo";

// ── history budget ────────────────────────────────────────────────────
const MAX_CLIENT_CONTEXT_CHARS = 48_000;

// How many attachments one turn analyzes. Each file is a full extract→enrich
// round-trip, so an unbounded loop would hang the turn for minutes; anything past
// this is NAMED as unread in the narration rather than dropped silently
// (operator ask 2026-08-04, k65).
const MAX_TURN_ATTACHMENTS = 5;

function estimateMessageChars(message: ChatMessage): number {
  return message.role.length + message.content.length + 16;
}

function buildMessagesFromChats(
  previousChats: ChatEntry[],
  nextPrompt: string,
): ChatMessage[] {
  const systemMessage: ChatMessage = {
    role: "system",
    content:
      "You are hugpy's media-intelligence assistant — warm, articulate, and genuinely " +
      "helpful. Speak naturally and conversationally, never robotically or in canned " +
      "phrases. You can analyze audio, images, and text. Use prior context when relevant, " +
      "but prioritize the user's latest message. " +
      // operator ask 2026-08-04 (k64): a file with no instruction is a question,
      // not a mandate — a default guessed here is a promise the user never made.
      "When the user gives you a file without saying what they want done with it, ask " +
      "what they'd like rather than assuming a default.",
  };

  const newestFirst: ChatMessage[] = [
    { role: "user", content: nextPrompt },
  ];

  for (let i = previousChats.length - 1; i >= 0; i -= 1) {
    const chat = previousChats[i];

    if ((chat.response as string)?.trim() && chat.status === "complete") {
      newestFirst.push({
        role: "assistant",
        content: (chat.response as string).trim(),
      });
    }

    if (chat.query?.trim()) {
      newestFirst.push({ role: "user", content: chat.query.trim() });
    }
  }

  const keptReversed: ChatMessage[] = [];
  let usedChars = estimateMessageChars(systemMessage);

  for (const message of newestFirst) {
    const cost = estimateMessageChars(message);
    if (usedChars + cost > MAX_CLIENT_CONTEXT_CHARS) break;
    keptReversed.push(message);
    usedChars += cost;
  }

  return [systemMessage, ...keptReversed.reverse()];
}

// ── component ─────────────────────────────────────────────────────────
export default function HugpyChat(): JSX.Element {
  const location = useLocation();
  const navigate = useNavigate();
  const mediaView = /\/media\/?$/i.test(location.pathname);

  const [draftPrompt, setDraftPrompt] = useState("");
  // Tools pre-selected in the ToolTray (by spec.key). They run on a normal send
  // (overriding the auto-router) and stay selected across sends until toggled off.
  const [selectedToolKeys, setSelectedToolKeys] = useState<Set<string>>(new Set());
  // Resolved once: the canned-demo worked example, or null (live / normal mode).
  const demoInit = useMemo(() => demoSeed(), []);
  const [chats, setChats] = useState<ChatEntry[]>(() => demoInit?.chats ?? []);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);

  // Persisted conversation history (localStorage; the media arm has no server
  // history endpoint). `conversationId` identifies the active thread.
  const [conversations, setConversations] = useState<Conversation[]>(
    () => demoInit?.conversations ?? loadConversations(),
  );
  const [conversationId, setConversationId] = useState<string>(
    () => demoInit?.conversationId ?? createChatId(),
  );

  // File attachments: uploaded refs + detected category + the intermediary's
  // parallel suggestion (see attachFiles / AttachmentBar).
  const [attached, setAttached] = useState<UploadedFileRef[]>([]);
  const [attachedKind, setAttachedKind] = useState<MediaCategory>("text");
  const [suggestion, setSuggestion] = useState("");
  const [suggesting, setSuggesting] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [attachError, setAttachError] = useState("");

  // Session file library: every upload/drop accumulates here as a durable,
  // viewable object (not cleared on submit). Shown in-thread + sidebar; the
  // open file id drives the inline preview modal.
  const [files, setFiles] = useState<StoredFile[]>(() => loadFiles());
  const [previewId, setPreviewId] = useState<string | null>(null);
  // Persist the Files library (metadata + saved transcripts) across reloads, the
  // same way Recents persists — so transcribed audio stays available in Files.
  useEffect(() => {
    saveFiles(files);
  }, [files]);

  const [modelOptions, setModelOptions] = useState<ChatModelOption[]>(
    () => getChatModelOptions(),
  );
  const [modelKey, setModelKey] = useState(DEFAULT_MODEL);
  const [maxNewTokens, setMaxNewTokens] = useState(512);

  // Populate the model dropdown from the live hugpy registry on mount.
  // Defaults to the first fetched model; falls back to the built-in list
  // (already in state) if the fetch fails, so the UI never renders empty.
  useEffect(() => {
    const controller = new AbortController();

    fetchChatModelOptions(controller.signal).then((options) => {
      if (controller.signal.aborted || !options.length) return;
      setModelOptions(options);
      setModelKey((current) =>
        options.some((option) => option.key === current)
          ? current
          : options[0].key,
      );
    });

    return () => controller.abort();
  }, []);
  const [temperature, setTemperature] = useState(0.2);
  const [topP, setTopP] = useState(0.95);
  const [streamResponse, setStreamResponse] = useState(true);

  const [sidebarCollapsed, setSidebarCollapsed] = useState(
    () => typeof window !== "undefined" && window.innerWidth < 768,
  );

  // Which view fills the main area: the chat shell (default) or the classic
  // tool console. Toggled from the sidebar; the chat thread is preserved.
  const [view, setView] = useState<"chat" | "console">("chat");

  const abortRef = useRef<AbortController | null>(null);
  const loading = activeChatId !== null;
  const isEmpty = chats.length === 0;

  // Persist the active conversation whenever it settles (not mid-stream) so
  // Recents/Search survive reloads. New, still-empty chats aren't saved.
  // The canned demo stays in-memory only — it must never write its sample
  // thread into a real visitor's localStorage history.
  useEffect(() => {
    if (demoInit || loading || chats.length === 0) return;
    const conv: Conversation = {
      id: conversationId,
      title: deriveTitle(chats),
      updatedAt: nowIso(),
      messages: chats,
    };
    setConversations((prev) => {
      const next = upsertConversation(prev, conv);
      saveConversations(next);
      return next;
    });
  }, [chats, loading, conversationId]);

  const modelOption = useMemo(
    () => getChatModelOption(modelKey),
    [modelKey, modelOptions],
  );
  const modelLabel = useMemo(
    () => getChatModelLabel(modelKey),
    [modelKey, modelOptions],
  );

  function patchChat(id: string, patch: Partial<ChatEntry>): void {
    setChats((prev) =>
      prev.map((chat) => (chat.id === id ? { ...chat, ...patch } : chat)),
    );
  }

  // ── file attachments ───────────────────────────────────────────────────
  // Build a tool input from the composer: a file input (refs → selectedIds) when
  // something is attached, else plain text.
  // Detect a link the user typed/pasted so URL tools (Read Webpage → /ml/fetch)
  // get it. Explicit http(s):// wins; a bare "example.com/…" token (the WHOLE
  // message) is promoted to https. Conservative on bare domains to avoid treating
  // prose like "see foo.bar" as a URL.
  function firstUrl(text: string): string | null {
    const t = (text || "").trim();
    const m = t.match(/\bhttps?:\/\/[^\s<>"']+/i);
    if (m) return m[0];
    if (/^(?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)+(?:\/[^\s]*)?$/i.test(t)) return "https://" + t;
    return null;
  }

  function buildToolInput(text: string, refs: UploadedFileRef[] = attached): MediaInputValue {
    if (refs.length === 0) {
      return { inputMode: "text", text, url: firstUrl(text) ?? "", files: [], uploadedFiles: [], selectedIds: [] } as never;
    }
    return {
      inputMode: "file",
      text,
      url: "",
      files: [],
      uploadedFiles: refs,
      selectedIds: refs.map((f) => f.id),
    } as never;
  }

  // The "active" file for a tool run: the live composer attachment if any, else
  // the file the user is referencing from the session library — the previewed
  // file, else the most recent upload (optionally preferring a category, e.g.
  // "audio" for transcribe). This is what lets a SELECTED tool (e.g. the Audio
  // Transcription button) run on an already-uploaded file instead of the model
  // replying "please upload a file" for a file that is already there.
  function resolveActiveRefs(preferKind?: string): UploadedFileRef[] {
    if (attached.length) return attached;
    let pick: StoredFile | undefined;
    if (preferKind) {
      for (let i = files.length - 1; i >= 0; i--) {
        if (String(files[i].kind) === preferKind) { pick = files[i]; break; }
      }
    }
    if (!pick && previewId) pick = files.find((f) => f.id === previewId);
    if (!pick) pick = files[files.length - 1];
    return pick ? [{ id: pick.id, name: pick.name, type: pick.type }] : [];
  }

  // Tools that consume an uploaded file of this category (transcribe⇐audio,
  // vision⇐image). Text/pdf/video have no file-consuming tool yet → [].
  function fileToolsFor(kind: MediaCategory): PageSpec[] {
    if (kind === "text") return [];
    return listPages().filter(
      (p) =>
        (p.accepts ?? []).includes(kind) &&
        (p.fields ?? []).some((f) => f.source === "selectedIds"),
    );
  }

  // What we can offer for each category when the user attaches a file and says
  // NOTHING (operator ask 2026-08-04, k64). Phrased as verb clauses rather than
  // reusing the chip TITLES verbatim ("Audio Transcription") so the question
  // reads as a sentence; the chips themselves stay the click path and are
  // pointed at explicitly, so the two can't offer different things.
  const CATEGORY_ACTIONS: Record<MediaCategory, string[]> = {
    image: ["describe what's in it", "pull out any text it contains", "answer questions about it"],
    audio: ["transcribe it", "summarize what's said", "answer questions about it"],
    video: ["transcribe the audio", "summarize what's said", "answer questions about it"],
    pdf: ["summarize it", "pull out the key topics", "answer questions about it"],
    document: ["summarize it", "pull out the key topics", "answer questions about it"],
    text: ["summarize it", "pull out the key topics", "answer questions about it"],
  };

  // The deterministic ASK for a bare attachment. Canned-but-warm and parametrized
  // by category — NOT model-generated: this message must be correct every time,
  // and an LLM asked to "ask a question" will sometimes answer instead.
  function attachmentAskText(refs: UploadedFileRef[], kind: MediaCategory): string {
    const subject =
      refs.length === 1 ? `“${refs[0].name}”` : `those ${refs.length} files`;
    const actions = CATEGORY_ACTIONS[kind] ?? CATEGORY_ACTIONS.text;
    const list =
      actions.length > 1
        ? `${actions.slice(0, -1).join(", ")}, or ${actions[actions.length - 1]}`
        : actions[0];
    const chips = fileToolsFor(kind).length > 0
      ? " The buttons just above the message box run those directly."
      : "";
    return (
      `I've got ${subject} — what would you like me to do with it? ` +
      `I can ${list}. Just tell me, and I'll get to it.${chips}`
    );
  }

  // Post the ask as its own completed assistant turn. The attachment is NOT
  // cleared: the chips stay live and a typed reply proceeds with the file still
  // bound, exactly as if the user had typed it in the first place.
  function askWhatToDoWithAttachment(): void {
    const refs = attached;
    const kind = attachedKind;
    setChats((prev) => [
      ...prev,
      {
        kind: "chat",
        id: createChatId(),
        // The user issued no instruction, so the turn is labelled with what they
        // actually did — the filename — never a synthesized "Analyze <file>".
        query: refs.map((r) => r.name).join(", "),
        response: attachmentAskText(refs, kind),
        status: "complete",
        finishReason: "stop",
        queryStartedAt: nowIso(),
        responseFinishedAt: nowIso(),
        modelKey,
        modelLabel: getChatModelLabel(modelKey),
        files: refs.map((r) => ({ id: r.id, name: r.name, kind })),
      },
    ]);
  }

  // Does this tool have the input it needs? File tools need an attached file;
  // text-consuming tools (summarize/keywords/embed) need substantive text;
  // generators (imagine/text-gen) need any prompt. This is the guard that stops
  // text routing to a file tool, and stops running a tool on a synthetic label.
  function toolCanRun(spec: PageSpec, input: MediaInputValue): boolean {
    const fields = spec.fields ?? [];
    if (fields.some((f) => f.source === "selectedIds")) {
      return Boolean(input.selectedIds && input.selectedIds[0]);
    }
    if (fields.some((f) => f.source === "url")) {
      return Boolean((input.url ?? "").trim());
    }
    const text = (input.text ?? "").trim();
    if (fields.some((f) => f.source === "text")) return text.length >= 12;
    if (fields.some((f) => f.name === "prompt")) return text.length > 0;
    return true;
  }

  function clearAttached(): void {
    setAttached([]);
    setAttachedKind("text");
    setSuggestion("");
    setSuggesting(false);
    setAttachError("");
  }

  // Parallel intuitive suggestion from the intermediary the moment a file lands.
  async function suggestForFile(refs: UploadedFileRef[], kind: MediaCategory): Promise<void> {
    setSuggesting(true);
    try {
      const tools = fileToolsFor(kind).map((p) => p.title).join(", ") || "answering questions about it";
      const system =
        `You are a media-intelligence assistant. A user just attached a ${kind} file ` +
        `named "${refs[0].name}". In ONE short, friendly sentence, suggest what you can ` +
        `do with it. Available analyses: ${tools}.`;
      const res = await generateChat({
        request_id: createChatId(),
        model_key: modelKey,
        messages: [
          { role: "system", content: system },
          { role: "user", content: "What can you do with this file?" },
        ],
        max_new_tokens: 60,
        temperature: 0.3,
        do_sample: true,
      });
      setSuggestion(typeof res.text === "string" ? res.text.trim() : "");
    } catch {
      setSuggestion("");
    } finally {
      setSuggesting(false);
    }
  }

  // Documents get full-text extracted in the background so the file library is
  // searchable by content (not just name). Operator-session-authed in-browser.
  async function extractForSearch(file: StoredFile): Promise<void> {
    if (file.kind !== "pdf" && file.kind !== "document" && file.kind !== "text") return;
    const spec = getPage("ml/extract");
    if (!spec) return;
    try {
      const input = {
        inputMode: "file", text: "", url: "",
        files: [], uploadedFiles: [], selectedIds: [file.id],
      } as never;
      const res = await dispatchTool(spec, input);
      if (res.ok) {
        const data = okValue(res) as { text?: string };
        const text = typeof data?.text === "string" ? data.text : "";
        setFiles((prev) => prev.map((f) =>
          f.id === file.id
            ? { ...f, extractedText: text, extractStatus: text ? "done" : "unsupported" }
            : f));
      } else {
        setFiles((prev) => prev.map((f) =>
          f.id === file.id ? { ...f, extractStatus: "error" } : f));
      }
    } catch {
      setFiles((prev) => prev.map((f) =>
        f.id === file.id ? { ...f, extractStatus: "error" } : f));
    }
  }

  // Upload dropped/browsed files, persist them to the session library (blob URL
  // for inline preview + async doc extraction), detect the category, suggest.
  async function attachFiles(incoming: File[]): Promise<void> {
    setUploading(true);
    setSuggestion("");
    setAttachError("");
    try {
      const refs = await uploadAttachments(incoming);
      if (!refs.length) throw new Error("no files accepted by the server");
      const kind = detectCategory(refs[0]);
      setAttached(refs);
      setAttachedKind(kind);

      // Persist to the library (dedupe by id) + start background extraction.
      const stored: StoredFile[] = refs.map((ref, i) => {
        const original = incoming[i];
        const fkind = detectCategory(ref);
        return {
          id: ref.id,
          name: ref.name,
          type: ref.type || original?.type || "",
          kind: fkind,
          size: original?.size,
          uploadedAt: nowIso(),
          blobUrl: original ? URL.createObjectURL(original) : undefined,
          extractStatus: (fkind === "pdf" || fkind === "document" || fkind === "text") ? "extracting" : "idle",
        };
      });
      setFiles((prev) => {
        const byId = new Map(prev.map((f) => [f.id, f]));
        for (const s of stored) if (!byId.has(s.id)) byId.set(s.id, s);
        return Array.from(byId.values());
      });
      stored.forEach((s) => { void extractForSearch(s); });

      void suggestForFile(refs, kind);
    } catch (e) {
      // Surface, don't hide — keep the bar up with the reason.
      setAttached([]);
      setAttachedKind("text");
      setSuggestion("");
      setSuggesting(false);
      setAttachError(e instanceof Error ? e.message : String(e));
    } finally {
      setUploading(false);
    }
  }

  // Run a category option (AttachmentBar chip) on the attached file, then narrate.
  function runFileOption(spec: PageSpec): void {
    const input = buildToolInput(draftPrompt.trim());
    clearAttached();
    void runToolFromTray(spec, input);
  }

  // Authoritative-intermediary system prefix (keeps results narrated, never raw).
  const NARRATION_SYS =
    "You are an authoritative media-intelligence assistant. You just performed " +
    "an analysis for the user's request; its output is below. Present and explain " +
    "it clearly and confidently in your own words, as if you did the work. Do not " +
    "mention tools, endpoints, or raw JSON.\n\nAnalysis output:\n";

  // Image-producing tools (text-to-image) render their result as a visual tool
  // turn so the user SEES the image rather than a text description of it.
  function resultHasImages(value: unknown): boolean {
    return (
      typeof value === "object" &&
      value !== null &&
      Array.isArray((value as { images?: unknown }).images)
    );
  }
  function producesVisual(spec: PageSpec, value: unknown): boolean {
    const produces = Array.isArray(spec.produces) ? spec.produces : [spec.produces];
    return produces.includes("imagegen") || resultHasImages(value);
  }
  function pushToolTurn(spec: PageSpec, result: unknown, operation: string): void {
    setChats((prev) => [
      ...prev,
      {
        kind: "tool",
        id: createChatId(),
        specKey: spec.key,
        title: spec.title,
        operation,
        result,
        status: "complete",
        queryStartedAt: nowIso(),
        responseFinishedAt: nowIso(),
      },
    ]);
  }

  // Run a chain (multi-step pipeline) in the chat: each step is dispatched to its
  // real endpoint via dispatchTool, results flow step→step, and the per-step
  // panel is shown as a tool turn (ChainOutput). Returns narration context.
  async function runChainStep(
    spec: PageSpec,
    input: MediaInputValue,
    id: string,
    controller: AbortController,
  ): Promise<string> {
    const chainKey = spec.key.slice("chain:".length);
    const results = await runChain(
      chainKey,
      input,
      async (pageKey, stepInput) => {
        const stepSpec = getPage(pageKey);
        patchChat(id, { status: "thinking", thinking: `Running ${stepSpec.title}…` });
        const res = await dispatchTool(stepSpec, stepInput, controller.signal);
        if (!res.ok) {
          const err = errorOf(res) as unknown;
          throw new Error(
            typeof err === "string"
              ? err
              : (err as { message?: string })?.message ?? "step failed",
          );
        }
        return okValue(res);
      },
    );
    patchChat(id, { toolUsed: spec.title });
    pushToolTurn(spec, results, "chain");

    const failed = results.find((r) => !r.ok);
    if (failed) {
      return `(The "${spec.title}" pipeline stopped: ${failed.error ?? "a step failed"}. Briefly tell the user which part didn't work.)`;
    }
    const lastOk = [...results].reverse().find((r) => r.ok);
    const finalText = lastOk ? extractText(lastOk.data) : "";
    return finalText
      ? `You ran the "${spec.title}" pipeline. Final output:\n${finalText.slice(0, 2500)}\n\nPresent this to the user clearly in your own words.`
      : `(The "${spec.title}" pipeline ran but produced no readable text; briefly tell the user.)`;
  }

  // Run ONE tool on a media input; return narration context (serialized result,
  // or a graceful note on failure). Marks the turn with the analysis it ran.
  async function runToolStep(
    spec: PageSpec,
    input: MediaInputValue,
    id: string,
    controller: AbortController,
  ): Promise<string> {
    patchChat(id, {
      status: "thinking",
      thinking: `Running ${spec.title}…`,
    });
    try {
      // Chains run their steps through runChain (each step dispatched to its real
      // endpoint); the synthetic /__chain/* path is never POSTed directly.
      if (spec.key.startsWith("chain:")) {
        return await runChainStep(spec, input, id, controller);
      }
      // Flat dispatch to the named endpoint; the backend's execute_prompt routes it.
      const res = await dispatchTool(spec, input, controller.signal);
      if (res.ok) {
        const value = okValue(res);
        patchChat(id, { toolUsed: spec.title }); // tag attribution only on a real result
        // Visual tools (image generation): show the image as its own tool turn,
        // and hand the narrator a short lead-in instead of raw JSON.
        if (producesVisual(spec, value)) {
          pushToolTurn(spec, value, "imagegen");
          return "(You generated an image from the user's prompt and displayed it to them above. In ONE short, friendly sentence, present it — do not describe pixels, dimensions, or mention tools.)";
        }
        // Structured ML tools (transcribe, summarize, keywords, embeddings/metadata,
        // extract, …): render the result PANEL directly via ExecutionOutput — the
        // showroom shows each tool's output this way, and the chat model is
        // unreliable at reproducing structured output. Hand the narrator only a
        // short lead-in. (Visual/imagegen is handled above; chat/text-gen below.)
        const produces = String(
          (Array.isArray(spec.produces) ? spec.produces[0] : spec.produces) ?? "",
        );
        const STRUCTURED_OUTPUT = new Set([
          "transcribe", "summarize", "keywords", "metadata", "extract", "intelligence", "analyze",
        ]);
        const isTranscribe = produces === "transcribe" || /transcrib/i.test(spec.key);
        if (STRUCTURED_OUTPUT.has(produces) || isTranscribe) {
          pushToolTurn(spec, value, produces || "transcribe");
          // Persist a transcript so the user can then ASK QUESTIONS about it later —
          // narrate() feeds a file's saved content into the model as context.
          if (isTranscribe) {
            const fid = (input as MediaInputValue)?.selectedIds?.[0];
            const transcript = serializeToolResult(value, spec).trim();
            if (transcript && fid) {
              setFiles((prev) =>
                prev.map((f) =>
                  f.id === fid ? { ...f, extractedText: transcript, extractStatus: "done" } : f,
                ),
              );
            }
          }
          const out = serializeToolResult(value, spec).trim();
          return out
            ? `(You ran ${spec.title} on the user's input and displayed the result to them above. In ONE short sentence, tell them it's ready. Do NOT repeat the output or ask them to upload anything.)`
            : `(${spec.title} ran but produced no result. Briefly tell the user, in one sentence.)`;
        }
        return serializeToolResult(value, spec);
      }
      const err = errorOf(res) as unknown;
      const msg = typeof err === "string" ? err : (err as { message?: string })?.message ?? "error";
      return `(the ${spec.title} step could not complete: ${msg})`;
    } catch (e) {
      return `(the ${spec.title} step could not complete: ${e instanceof Error ? e.message : String(e)})`;
    }
  }

  // Stream the authoritative answer into turn `id`, grounded in toolContext when
  // a tool ran. Shared by the typed flow, the tray, and file options.
  async function narrate(
    id: string,
    userText: string,
    toolContext: string,
    requestModelKey: string,
    requestMaxNewTokens: number,
    controller: AbortController,
    historyChats: ChatEntry[] = chats,
    activeFile?: { name: string; kind: string; content?: string } | null,
  ): Promise<void> {
    // Self-resolve the active file when the caller didn't pass one, so EVERY
    // narration path (tool tray, typed flow, file options) is file-aware — not
    // just the main submit. Pass null explicitly to opt out.
    if (activeFile === undefined) {
      const refs = resolveActiveRefs();
      const sf = refs[0] ? files.find((f) => f.id === refs[0].id) : undefined;
      activeFile = refs[0]
        ? { name: refs[0].name, kind: String(sf?.kind ?? attachedKind), content: sf?.extractedText?.trim() || undefined }
        : null;
    }
    const messages: ChatMessage[] = buildMessagesFromChats(historyChats, userText);
    const sysParts: string[] = [];
    // File awareness: the chat model can't see the session library on its own, so
    // tell it a file is already uploaded. Without this it replies "please upload a
    // file" for a file the user already attached.
    if (activeFile) {
      let note =
        `The user has already uploaded a ${activeFile.kind} file named "${activeFile.name}". ` +
        `It is attached and available to you — never ask the user to upload or provide a file, ` +
        `and never claim no file was provided.`;
      if (activeFile.content && activeFile.content.trim()) {
        const label = activeFile.kind === "audio" || activeFile.kind === "video" ? "transcript" : "content";
        note +=
          ` Use the ${label} below to answer the user's questions about "${activeFile.name}" ` +
          `(summarize it, quote it, explain it — whatever they ask).` +
          `\n\n--- ${label} of "${activeFile.name}" ---\n${activeFile.content.trim()}`;
      }
      sysParts.push(note);
    }
    if (toolContext) sysParts.push(NARRATION_SYS + toolContext);
    if (sysParts.length) {
      messages.unshift({ role: "system", content: sysParts.join("\n\n") });
    }
    patchChat(id, {
      status: streamResponse ? "thinking" : "queued",
      thinking: streamResponse ? "Composing…" : "Generating…",
    });
    const request: ChatRequest = {
      request_id: id,
      model_key: requestModelKey,
      messages,
      max_new_tokens: requestMaxNewTokens,
      temperature,
      top_p: topP,
      do_sample: temperature > 0,
    };
    if (streamResponse) await runStreamingChat(id, request, controller);
    else await runNonStreamingChat(id, request, controller);
  }

  // Tray / category-option entry point: explicitly run `spec` (tool chosen by the
  // user, not the router) on the current input, then narrate — same path as typing.
  async function runToolFromTray(
    spec: PageSpec,
    input?: MediaInputValue,
  ): Promise<void> {
    if (loading) return;
    const draft = draftPrompt.trim();
    // The tool runs on REAL input (draft text / attached file) — never the label.
    const toolInput: MediaInputValue = input ?? buildToolInput(draft);

    // Nothing to work on → nudge for what the tool needs instead of running on nothing.
    if (!toolCanRun(spec, toolInput)) {
      const fileKind = (spec.accepts ?? [])[0] ?? "file";
      const needsFile = (spec.fields ?? []).some((f) => f.source === "selectedIds");
      const nudgeId = createChatId();
      setChats((prev) => [
        ...prev,
        {
          kind: "chat",
          id: nudgeId,
          query: spec.title,
          response: needsFile
            ? `Add a ${fileKind} file (the ＋ button or drag-and-drop), then I'll run ${spec.title} on it.`
            : `Type or paste the text you'd like me to run ${spec.title} on.`,
          status: "complete",
          finishReason: "stop",
          queryStartedAt: nowIso(),
          responseFinishedAt: nowIso(),
          modelKey,
          modelLabel: getChatModelLabel(modelKey),
        },
      ]);
      return;
    }

    const id = createChatId();
    const userText = draft || `Run ${spec.title}`;
    const requestModelKey = modelKey;
    const turnFiles = (toolInput?.selectedIds ?? [])
      .map((fid) => files.find((f) => f.id === fid))
      .filter((f): f is StoredFile => Boolean(f))
      .map((f) => ({ id: f.id, name: f.name, kind: f.kind }));
    setActiveChatId(id);
    setChats((prev) => [
      ...prev,
      {
        kind: "chat",
        id,
        query: userText,
        response: "",
        status: "thinking",
        thinking: `Running ${spec.title}…`,
        queryStartedAt: nowIso(),
        modelKey: requestModelKey,
        modelLabel: getChatModelLabel(requestModelKey),
        files: turnFiles.length ? turnFiles : undefined,
      },
    ]);
    setDraftPrompt("");
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const ctx = await runToolStep(spec, toolInput, id, controller);
      await narrate(
        id,
        userText,
        ctx,
        requestModelKey,
        Math.min(maxNewTokens, modelOption.maxNewTokens),
        controller,
      );
    } catch (error) {
      patchChat(id, {
        status: controller.signal.aborted ? "cancelled" : "error",
        thinking: undefined,
        response: controller.signal.aborted
          ? undefined
          : error instanceof Error
            ? `Error: ${error.message}`
            : `Error: ${String(error)}`,
        finishReason: controller.signal.aborted ? "cancelled" : undefined,
        responseFinishedAt: nowIso(),
      });
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
      setActiveChatId((current) => (current === id ? null : current));
    }
  }

  function useMaxTokens() {
    setMaxNewTokens(modelOption.maxNewTokens);
  }

  function changeModelKey(nextModelKey: string) {
    const nextModel = getChatModelOption(nextModelKey);
    setModelKey(nextModelKey);
    setMaxNewTokens(nextModel.defaultMaxNewTokens);
  }

  async function submitPrompt(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const formData = new FormData(event.currentTarget);
    const submittedPrompt = String(
      formData.get("prompt-textarea") ?? "",
    ).trim();

    // A file alone (no text) is still a valid submit — but it is a QUESTION, not
    // an instruction. Ask what they want instead of synthesizing "Analyze <file>"
    // and running the whole pipeline on a guess (operator ask 2026-08-04, k64:
    // "if no direction is given, ask what the user would like done with them
    // rather than spitting out defaults"). A tray tool toggled on IS direction,
    // as is any typed text however vague ("look at this") — only EMPTY text with
    // no selected tool triggers the ask. The attachment stays attached.
    if ((!submittedPrompt && attached.length === 0) || loading) return;
    if (!submittedPrompt && attached.length > 0 && selectedToolKeys.size === 0) {
      askWhatToDoWithAttachment();
      return;
    }

    const id = createChatId();
    const queryStartedAt = nowIso();
    const requestModelKey = modelKey;
    const requestModelLabel = getChatModelLabel(requestModelKey);
    const requestMaxNewTokens = Math.min(
      maxNewTokens,
      modelOption.maxNewTokens,
    );

    // What the turn displays / the narrator answers. A bare attachment no longer
    // reaches here without a selected tool (it gets the ask above), so the only
    // remaining empty-text case is "tool toggled + file, no words" — label that
    // with the filename rather than a synthesized instruction (k64).
    const effectivePrompt =
      submittedPrompt ||
      (attached.length ? attached.map((a) => a.name).join(", ") : "");

    setDraftPrompt("");
    setActiveChatId(id);

    setChats((prev) => [
      ...prev,
      {
        kind: "chat",
        id,
        query: effectivePrompt,
        response: "",
        status: streamResponse ? "thinking" : "queued",
        thinking: streamResponse
          ? `Streaming with ${requestModelLabel}…`
          : `Generating with ${requestModelLabel}…`,
        queryStartedAt,
        modelKey: requestModelKey,
        modelLabel: requestModelLabel,
        files: attached.length
          ? attached.map((a) => ({ id: a.id, name: a.name, kind: attachedKind }))
          : undefined,
      },
    ]);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      let toolContext = "";
      if (selectedToolKeys.size > 0) {
        // ── Explicit tool/chain selection (tray toggles) WINS over the auto
        // pipelines — run the selected tools on this turn's input (text and/or
        // the active file). A file-consuming tool (e.g. Audio Transcription)
        // resolves its file from the composer OR the session library, so a tool
        // selected after upload runs on the uploaded file. Unrunnable → skipped.
        const parts: string[] = [];
        for (const key of selectedToolKeys) {
          const spec = getPage(key);
          if (!spec) continue;
          const needsFile = (spec.fields ?? []).some((f) => f.source === "selectedIds");
          const refs = needsFile ? resolveActiveRefs(spec.accepts?.[0]) : attached;
          // k65 — the same one-file-only drop the intelligence path had: a
          // file-consuming tool ran on refs[0] and the rest went unmentioned. The
          // /ml amenities take one file per call, so run the tool once PER FILE
          // (sequential, cancellable, capped) — each gets its own tool turn.
          const runs = needsFile && refs.length > 1
            ? refs.slice(0, MAX_TURN_ATTACHMENTS).map((r) => [r])
            : [refs];
          for (const group of runs) {
            if (controller.signal.aborted) break;
            const selInput = buildToolInput(submittedPrompt, group);
            if (toolCanRun(spec, selInput)) {
              parts.push(await runToolStep(spec, selInput as never, id, controller));
            }
          }
          if (needsFile && refs.length > MAX_TURN_ATTACHMENTS) {
            const names = refs.slice(MAX_TURN_ATTACHMENTS).map((r) => r.name).join(", ");
            parts.push(
              `(${spec.title} ran on the first ${MAX_TURN_ATTACHMENTS} files only. NOT ` +
              `processed: ${names}. Tell the user plainly which files were left out.)`,
            );
          }
        }
        clearAttached();
        toolContext = parts.filter(Boolean).join("\n\n");
      }
      // ── Attachment → media_intelligence bridge ───────────────────────────
      // The "missing medium": instead of deriving ONE tool, run the full pipeline
      // (extract → enrich) and render a structured DocumentIntelligence panel, with
      // a short conversational lead-in narrated into this turn.
      else if (attached.length) {
        // EVERY attachment is analyzed, not just the first (operator ask
        // 2026-08-04, k65 — the rest used to be dropped in silence). Sequentially,
        // because each file is its own /ml round-trip and the GPU amenities
        // serialize anyway; the shared AbortController stops the queue on cancel.
        const runRefs = attached.slice(0, MAX_TURN_ATTACHMENTS);
        const skipped = attached.slice(MAX_TURN_ATTACHMENTS);
        clearAttached();

        const parts: string[] = [];
        const failureLines: string[] = [];
        let analyzed = 0;

        for (let i = 0; i < runRefs.length; i += 1) {
          if (controller.signal.aborted) break;
          const ref = runRefs[i];
          // Per-file kind: a mixed selection (a PDF and a recording) must not all
          // be treated as whatever the FIRST file happened to be.
          const kind = detectCategory(ref);
          const label = runRefs.length > 1
            ? `file ${i + 1}/${runRefs.length}: ${ref.name}`
            : "";

          if (!supportsIntelligence(kind)) {
            // No extractor for this kind yet (e.g. plain text attachments): stay graceful.
            parts.push(
              `(${label ? `[${label}] ` : ""}The user attached a ${kind} file ` +
              `"${ref.name}", which the media-intelligence pipeline does not handle ` +
              `yet. Briefly say so — suggest a PDF, document, image, audio, or video ` +
              `file — then answer the rest of their message.)`,
            );
            continue;
          }

          patchChat(id, {
            status: "thinking",
            thinking: runRefs.length > 1
              ? `Analyzing ${ref.name} (${i + 1}/${runRefs.length})…`
              : `Analyzing ${ref.name}…`,
          });
          const di = await runDocumentIntelligence(
            ref,
            kind,
            submittedPrompt,
            controller.signal,
          );
          // A failed read must be UNMISTAKABLE regardless of what the narrator
          // says about it (operator ask 2026-08-04, k64): the LLM note below is
          // a soft suggestion, so the stage's own reason — the backend's honest
          // sentence, e.g. "legacy binary .doc isn't supported — save it as
          // .docx and re-attach" — is surfaced DETERMINISTICALLY in two places
          // the model cannot soften: the tool turn's error line, and the
          // attachment bar's banner above the composer.
          const reason = di.ok ? "" : failureReasonFor(di);
          const failureLine = reason
            ? `Couldn't read "${ref.name}": ${reason}`
            : `Couldn't analyze ${kind} file "${ref.name}".`;
          // Structured panel as its own tool turn (ToolRunTurn → ExecutionOutput),
          // one PER FILE so every attachment's result is inspectable on its own.
          const toolId = createChatId();
          setChats((prev) => [
            ...prev,
            {
              kind: "tool",
              id: toolId,
              specKey: "media/intelligence",
              title: `Document intelligence · ${ref.name}`,
              operation: "intelligence",
              result: di,
              status: di.ok ? "complete" : "error",
              error: di.ok ? undefined : failureLine,
              queryStartedAt: nowIso(),
              responseFinishedAt: nowIso(),
            },
          ]);
          if (di.ok) analyzed += 1;
          else failureLines.push(failureLine);
          parts.push(
            di.ok
              ? narrationContextFor(di, { label, instruct: runRefs.length === 1 })
              : `(${label ? `[${label}] ` : ""}The media-intelligence pipeline could ` +
                `not read the ${kind} file "${ref.name}". The exact reason, already ` +
                `shown to the user, is: ${reason || "no reason was reported"}. Say that ` +
                `reason back to them in your own words — do NOT contradict it, invent a ` +
                `different cause, or claim you analyzed the file — then answer the rest ` +
                `of their message.)`,
          );
        }

        if (analyzed > 0) patchChat(id, { toolUsed: "Document intelligence" });
        // The banner is single-line, so many failures become one deterministic
        // sentence rather than an arbitrary pick — no failure goes unnamed.
        if (failureLines.length === 1) setAttachError(failureLines[0]);
        else if (failureLines.length > 1) {
          setAttachError(
            `${failureLines.length} of ${runRefs.length} files couldn't be read — ` +
            failureLines.join(" · "),
          );
        }
        // Never drop an attachment quietly: say which ones went unread and why.
        if (skipped.length) {
          const names = skipped.map((r) => r.name).join(", ");
          parts.push(
            `(Only the first ${MAX_TURN_ATTACHMENTS} attachments were analyzed this ` +
            `turn. NOT analyzed: ${names}. Tell the user plainly that these ${skipped.length} ` +
            `file(s) were not read and that they can send them in a follow-up message.)`,
          );
        }
        if (runRefs.length > 1) {
          parts.push(
            "\nGive a 1–2 sentence overview covering EACH file above (say which is which). " +
            "Do NOT list everything — the full breakdown per file is shown to the user separately.",
          );
        }
        toolContext = parts.filter(Boolean).join("\n\n");
      } else if (firstUrl(submittedPrompt)) {
        // ── A pasted/typed link (no attachment, no pre-selection) → read the page
        // and narrate from its text, same shape as the document pipeline. ──
        patchChat(id, { status: "thinking", thinking: "Reading the page…" });
        const spec = getPage("ml/fetch");
        const input = buildToolInput(submittedPrompt);
        if (spec && toolCanRun(spec, input)) {
          toolContext = await runToolStep(spec, input as never, id, controller);
        }
      } else {
        // ── No tool toggled, no fresh attachment → let the intermediary route.
        // If the user is referencing an already-uploaded library file, tell the
        // router (so it stops refusing file tools) and feed that file to a routed
        // file-consuming tool. ──
        patchChat(id, { status: "thinking", thinking: "Assessing…" });
        const activeRefs = resolveActiveRefs();
        const turnSF = activeRefs[0] ? files.find((f) => f.id === activeRefs[0].id) : undefined;
        // Already-extracted file (e.g. a transcript we saved): DON'T re-run an
        // extraction tool — just chat, with the content fed in as context by
        // narrate(). This is what makes "summarize it" / "output the lyrics" /
        // "what were they" answer from the existing transcript instead of
        // re-transcribing (or asking for the file again) on every turn.
        const alreadyExtracted = Boolean(turnSF?.extractedText && turnSF.extractedText.trim());
        const activeFile = activeRefs[0]
          ? { name: activeRefs[0].name, kind: String(turnSF?.kind ?? attachedKind) }
          : null;
        const route = alreadyExtracted
          ? null
          : await routeTool(
              submittedPrompt,
              requestModelKey,
              `route-${id}`,
              controller.signal,
              activeFile,
            );
        if (route) {
          const spec = getPage(route.specKey);
          const needsFile = (spec?.fields ?? []).some((f) => f.source === "selectedIds");
          const input = buildToolInput(submittedPrompt, needsFile ? resolveActiveRefs(spec?.accepts?.[0]) : attached);
          // Only run if the routed tool actually has its input — never summarize
          // gibberish or send text to a file tool with no file.
          if (spec && toolCanRun(spec, input)) {
            toolContext = await runToolStep(spec, input as never, id, controller);
          }
        }
      }
      // Resolve the file in play this turn (composer or referenced library file)
      // so the narrator always knows it exists — even when no tool ran.
      const turnRefs = resolveActiveRefs();
      const turnSF = turnRefs[0] ? files.find((f) => f.id === turnRefs[0].id) : undefined;
      const activeFileForTurn = turnRefs[0]
        ? {
            name: turnRefs[0].name,
            kind: String(turnSF?.kind ?? attachedKind),
            content: turnSF?.extractedText && turnSF.extractedText.trim() ? turnSF.extractedText : undefined,
          }
        : null;
      await narrate(
        id,
        effectivePrompt,
        toolContext,
        requestModelKey,
        requestMaxNewTokens,
        controller,
        chats,
        activeFileForTurn,
      );
    } catch (error) {
      if (controller.signal.aborted) {
        patchChat(id, {
          status: "cancelled",
          thinking: undefined,
          finishReason: "cancelled",
          responseFinishedAt: nowIso(),
        });
      } else {
        patchChat(id, {
          status: "error",
          thinking: undefined,
          response:
            error instanceof Error
              ? `Error: ${error.message}`
              : `Error: ${String(error)}`,
          responseFinishedAt: nowIso(),
        });
      }
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
      setActiveChatId((current) => (current === id ? null : current));
    }
  }

  async function runStreamingChat(
    id: string,
    request: ChatRequest,
    controller: AbortController,
  ) {
    for await (const evt of streamChat(request, controller.signal)) {
      if (evt.type === "token") {
        setChats((prev) =>
          prev.map((chat) =>
            chat.id === id
              ? {
                  ...chat,
                  status: "streaming",
                  thinking: undefined,
                  response:
                    typeof chat.response === "string"
                      ? chat.response + evt.text
                      : evt.text,
                }
              : chat,
          ),
        );
      }

      if (evt.type === "done") {
        patchChat(id, {
          status: evt.finish_reason === "cancelled" ? "cancelled" : "complete",
          finishReason: evt.finish_reason,
          thinking: undefined,
          responseFinishedAt: nowIso(),
        });
      }

      if (evt.type === "error") {
        patchChat(id, {
          status: "error",
          thinking: undefined,
          response: `Error: ${evt.message}`,
          responseFinishedAt: nowIso(),
        });
      }
    }
  }

  async function runNonStreamingChat(
    id: string,
    request: ChatRequest,
    controller: AbortController,
  ) {
    patchChat(id, {
      status: "thinking",
      thinking: "Generating full response…",
    });

    const data = await generateChat(request, controller.signal);

    patchChat(id, {
      status: "complete",
      thinking: undefined,
      response: data.text ?? "",
      finishReason: data.finish_reason ?? "stop",
      responseFinishedAt: nowIso(),
    });
  }

  function stopGeneration(): void {
    if (!activeChatId) return;
    const stoppedChatId = activeChatId;

    abortRef.current?.abort();
    if (streamResponse) void cancelChat(stoppedChatId);

    patchChat(stoppedChatId, {
      status: "cancelled",
      thinking: undefined,
      finishReason: "cancelled",
      responseFinishedAt: nowIso(),
    });

    setActiveChatId(null);
    abortRef.current = null;
  }

  // Re-run a turn's answer in place (regenerate / switch-model / edit). Uses the
  // history BEFORE this turn so its own stale answer never leaks into context,
  // and drops any tool panels that belonged to the previous run.
  async function regenerateTurn(
    chatId: string,
    opts?: { text?: string; modelKey?: string },
  ): Promise<void> {
    if (loading) return;
    const index = chats.findIndex((c) => c.id === chatId);
    if (index < 0) return;
    const target = chats[index];
    const userText = (opts?.text ?? target.query ?? "").trim();
    if (!userText) return;
    const requestModelKey = opts?.modelKey ?? target.modelKey ?? modelKey;
    const requestModelLabel = getChatModelLabel(requestModelKey);
    const requestMaxNewTokens = Math.min(
      maxNewTokens,
      getChatModelOption(requestModelKey).maxNewTokens,
    );
    const historyChats = chats.slice(0, index);

    // Reset the turn + remove its trailing tool panels, in one update.
    setChats((prev) => {
      const idx = prev.findIndex((c) => c.id === chatId);
      if (idx < 0) return prev;
      let end = idx + 1;
      while (end < prev.length && prev[end].kind === "tool") end += 1;
      const reset: ChatEntry = {
        ...prev[idx],
        query: userText,
        response: "",
        status: streamResponse ? "thinking" : "queued",
        thinking: "Assessing…",
        toolUsed: undefined,
        finishReason: undefined,
        responseFinishedAt: undefined,
        modelKey: requestModelKey,
        modelLabel: requestModelLabel,
      };
      return [...prev.slice(0, idx), reset, ...prev.slice(end)];
    });

    setActiveChatId(chatId);
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      let toolContext = "";
      const input = {
        inputMode: "text",
        text: userText,
        url: firstUrl(userText) ?? "",
        files: [],
        uploadedFiles: [],
        selectedIds: [],
      } as never;
      if (firstUrl(userText)) {
        patchChat(chatId, { status: "thinking", thinking: "Reading the page…" });
        const spec = getPage("ml/fetch");
        if (spec && toolCanRun(spec, input)) {
          toolContext = await runToolStep(spec, input, chatId, controller);
        }
      } else {
        const route = await routeTool(
          userText,
          requestModelKey,
          `route-${chatId}`,
          controller.signal,
        );
        if (route) {
          const spec = getPage(route.specKey);
          if (toolCanRun(spec, input)) {
            toolContext = await runToolStep(spec, input, chatId, controller);
          }
        }
      }
      await narrate(
        chatId,
        userText,
        toolContext,
        requestModelKey,
        requestMaxNewTokens,
        controller,
        historyChats,
      );
    } catch (error) {
      patchChat(chatId, {
        status: controller.signal.aborted ? "cancelled" : "error",
        thinking: undefined,
        response: controller.signal.aborted
          ? undefined
          : error instanceof Error
            ? `Error: ${error.message}`
            : `Error: ${String(error)}`,
        finishReason: controller.signal.aborted ? "cancelled" : undefined,
        responseFinishedAt: nowIso(),
      });
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
      setActiveChatId((cur) => (cur === chatId ? null : cur));
    }
  }

  function editTurn(chatId: string, newText: string): void {
    void regenerateTurn(chatId, { text: newText });
  }

  function deleteTurn(chatId: string): void {
    setChats((prev) => {
      const idx = prev.findIndex((c) => c.id === chatId);
      if (idx < 0) return prev;
      let end = idx + 1;
      while (end < prev.length && prev[end].kind === "tool") end += 1;
      return [...prev.slice(0, idx), ...prev.slice(end)];
    });
  }

  function startNewChat(): void {
    if (loading) abortRef.current?.abort();
    setChats([]);
    setActiveChatId(null);
    setDraftPrompt("");
    setConversationId(createChatId());
    setView("chat");
  }

  function selectConversation(id: string): void {
    if (id === conversationId) {
      setView("chat");
      return;
    }
    if (loading) {
      abortRef.current?.abort();
      setActiveChatId(null);
    }
    const conv = conversations.find((c) => c.id === id);
    if (!conv) return;
    setConversationId(id);
    setChats(conv.messages);
    setView("chat");
  }

  function deleteConversation(id: string): void {
    setConversations((prev) => {
      const next = prev.filter((c) => c.id !== id);
      saveConversations(next);
      return next;
    });
    if (id === conversationId) {
      setChats([]);
      setActiveChatId(null);
      setConversationId(createChatId());
    }
  }

  function clearAllConversations(): void {
    saveConversations([]);
    setConversations([]);
    setChats([]);
    setActiveChatId(null);
    setConversationId(createChatId());
  }

  // ── composer + controls + disclaimer assembly (shared between states) ──
  const composerBlock = (
    <>
      {/* File avenue: attached file → intermediary suggestion + category options. */}
      <AttachmentBar
        files={attached}
        kind={attachedKind}
        suggestion={suggestion}
        suggesting={suggesting}
        options={fileToolsFor(attachedKind)}
        error={attachError}
        onRemove={clearAttached}
        onRunOption={runFileOption}
      />
      <Composer
        prompt={draftPrompt}
        loading={loading}
        modelLabel={modelLabel}
        onPromptChange={setDraftPrompt}
        onSubmit={submitPrompt}
        onStop={stopGeneration}
        onAttachFiles={attachFiles}
        uploading={uploading}
        hasAttachment={attached.length > 0}
      />
      {/* General tool tray (also available alongside the LLM-led flow): pick any
          tool explicitly — its result is still narrated by the intermediary. */}
      <ToolTray
        selected={selectedToolKeys}
        onToggle={(spec) =>
          setSelectedToolKeys((prev) => {
            const next = new Set(prev);
            if (next.has(spec.key)) next.delete(spec.key);
            else next.add(spec.key);
            return next;
          })
        }
      />
      <ComposerControls
        loading={loading}
        modelKey={modelKey}
        modelOptions={modelOptions}
        maxNewTokens={maxNewTokens}
        maxAllowedTokens={modelOption.maxNewTokens}
        temperature={temperature}
        topP={topP}
        streamResponse={streamResponse}
        onModelKeyChange={changeModelKey}
        onMaxNewTokensChange={setMaxNewTokens}
        onTemperatureChange={setTemperature}
        onTopPChange={setTopP}
        onStreamResponseChange={setStreamResponse}
        onUseMaxTokens={useMaxTokens}
      />
      <Disclaimer />
    </>
  );

  return (
    <div className="hugpy-chat-scope flex w-full bg-token-main-surface-primary text-token-text-primary">
      <Sidebar
        collapsed={sidebarCollapsed}
        view={view}
        onToggle={() => setSidebarCollapsed((v) => !v)}
        onNewChat={startNewChat}
        onSelectChat={() => setView("chat")}
        onSelectConsole={() => setView("console")}
        files={files}
        onOpenFile={(id) => setPreviewId(id)}
        onDeleteFile={(id) => {
          void deleteUploadedFile(id); // remove from the server store (best-effort)
          setFiles((prev) => prev.filter((f) => f.id !== id));
          setPreviewId((cur) => (cur === id ? null : cur));
        }}
        conversations={conversations}
        activeConversationId={conversationId}
        onSelectConversation={selectConversation}
        onDeleteConversation={deleteConversation}
        onClearAllConversations={clearAllConversations}
      />

      {/* Mobile drawer backdrop: tap outside the off-canvas sidebar to close it. */}
      {!sidebarCollapsed && (
        <button
          type="button"
          aria-label="Close sidebar"
          onClick={() => setSidebarCollapsed(true)}
          className="fixed inset-0 z-30 bg-black/40 md:hidden"
        />
      )}

      <div className="relative flex min-h-0 min-w-0 w-full flex-1 flex-col">
        <ThreadHeader
          modelLabel={modelLabel}
          sidebarCollapsed={sidebarCollapsed}
          plainTitle="Media Intelligence"
          onReopenSidebar={() => setSidebarCollapsed(false)}
        />

        <main
          id="main"
          tabIndex={-1}
          className="flex min-h-0 flex-1 flex-col outline-none"
        >
          {view === "console" ? (
            /* ─── CONSOLE VIEW: the classic tool console (sidebar switch) ─ */
            <div className="flex min-h-0 flex-1 flex-col overflow-y-auto p-4">
              <HugpyConsole />
            </div>
          ) : isEmpty ? (
            /* ─── EMPTY STATE: centered headline + composer ───────── */
            <div className="flex flex-1 items-center justify-center px-4">
              <div
                className="
                  mx-auto w-full
                  max-w-[var(--thread-content-max-width-sm)]
                  lg:max-w-[var(--thread-content-max-width-lg)]
                  -mt-[calc(var(--header-height)/2)]
                "
              >
                <h1 className="mb-6 text-center text-[28px] font-normal tracking-[0.07px] text-token-text-primary">
                  What are you working on?
                </h1>
                {composerBlock}
              </div>
            </div>
          ) : (
            /* ─── LOADED STATE: thread above, docked composer below ── */
            <>
              <Thread
                chats={chats}
                loading={loading}
                modelLabel={modelLabel}
                models={modelOptions.map((o) => ({
                  key: o.key,
                  label: getChatModelLabel(o.key),
                }))}
                onEditQuery={editTurn}
                onRegenerate={(id, mk) => regenerateTurn(id, { modelKey: mk })}
                onDelete={deleteTurn}
                onOpenFile={(id) => setPreviewId(id)}
              />

              <div
                id="thread-bottom-container"
                className="sticky bottom-0 z-10 bg-token-main-surface-primary"
              >
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
                      mx-auto pt-2
                      max-w-[var(--thread-content-max-width-sm)]
                      lg:max-w-[var(--thread-content-max-width-lg)]
                    "
                  >
                    {composerBlock}
                  </div>
                </div>
              </div>
            </>
          )}
        </main>
      </div>

      <FilePreview
        file={files.find((f) => f.id === previewId) ?? null}
        onClose={() => setPreviewId(null)}
      />
    </div>
  );
}
