export type ChatStatus =
  | "queued"
  | "thinking"
  | "streaming"
  | "complete"
  | "cancelled"
  | "error";

// A thread entry is either a chat turn (kind "chat", the default) or a tool run
// (kind "tool"). Kept as ONE widened interface (chat + tool fields optional)
// rather than a strict discriminated union to minimise churn across existing
// consumers; the runtime narrows on `kind`.
export interface ToolSuggestion {
  specKey: string;          // a real registry key (validated via getPage)
  reason: string;
  prefillText?: string;
}

export interface ChatEntry {
  kind?: "chat" | "tool";   // default "chat"
  id: string;
  status: ChatStatus;
  // chat-turn fields
  query?: string;
  response?: string | string[];
  thinking?: string;
  queryStartedAt?: string;
  responseFinishedAt?: string;
  finishReason?: string;
  modelKey?: string;
  modelLabel?: string;
  toolUsed?: string;                // authoritative-intermediary: the analysis the LLM ran for this turn
  files?: { id: string; name: string; kind: string }[];  // attachments used in this turn (open in FilePreview)
  suggestions?: ToolSuggestion[];   // (legacy) inline LLM tool suggestions
  // tool-run fields (kind === "tool")
  specKey?: string;         // registry key of the tool that ran
  title?: string;           // tool display title
  operation?: string;       // PageSpec.produces — drives ExecutionOutput
  input?: unknown;          // MediaInputValue snapshot that was run
  result?: unknown;         // okValue(Result) on success
  error?: string;
  origin?: "tray" | "suggestion";
  restored?: boolean;       // loaded from history with its heavy result stripped
}

export interface ChatDisplayProps {
  chats: ChatEntry[];
  loading?: boolean;
  modelLabel: string;
  onEditQuery?: (chatId: string) => void;
}

export type ChatRole = "system" | "user" | "assistant";

export type ChatMessage = {
  role: ChatRole;
  content: string;
};

export interface ChatRequest {
  request_id: string;
  model_key?: string;
  messages: ChatMessage[];
  max_new_tokens?: number;
  temperature?: number;
  top_p?: number;
  do_sample?: boolean;
}

export type StreamEvent =
  | { type: "token"; request_id: string; text: string }
  | {
      type: "done";
      request_id: string;
      input_tokens: number;
      output_chunks: number;
      finish_reason: "stop" | "max_tokens" | "cancelled" | "error";
    }
  | { type: "error"; request_id: string; message: string };

export type GenerateChatResponse = {
  ok: boolean;
  request_id: string;
  text?: string;
  error?: string;
  finish_reason?: string;
};