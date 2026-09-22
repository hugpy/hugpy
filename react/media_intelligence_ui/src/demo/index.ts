// Public surface of the media-intelligence demo layer.
import { isCanned } from "./mode";
import { seedChats, seedConversations, seedConversationId } from "./fixtures";
import type { ChatEntry } from "../chat/src/imports";
import type { Conversation } from "../chat/src/utilities/chatHistory";

export interface DemoSeed {
  chats: ChatEntry[];
  conversations: Conversation[];
  conversationId: string;
}

/**
 * The canned worked example (earnings-call thread) used to seed HugpyChat on
 * first mount, so the CANNED demo opens already populated. Returns null in live
 * mode and in normal (non-demo) operation, leaving the chat empty as usual.
 */
export function demoSeed(): DemoSeed | null {
  if (!isCanned()) return null;
  return {
    chats: seedChats(),
    conversations: seedConversations(),
    conversationId: seedConversationId(),
  };
}

export {
  getDemoConfig,
  isDemo,
  isCanned,
  isLive,
  isEmbedded,
  demoUrl,
  type DemoMode,
  type DemoConfig,
} from "./mode";
export { installDemoTransport, uninstallDemoTransport } from "./demoFetch";
export { default as DemoBanner } from "./DemoBanner";
