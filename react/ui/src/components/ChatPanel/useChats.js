import { useCallback, useSyncExternalStore } from 'react'
import * as chatStore from './chatStore.js'

// Thin React binding over chatStore's module-level singleton (see
// chatStore.js for why the data lives there and not in this hook's own
// useState). useSyncExternalStore re-renders whichever component calls this
// whenever the store changes — including changes made while THIS component
// wasn't even mounted, e.g. tokens that streamed in during a route change and
// are only now being subscribed to again.
//
// Same exported shape as before ({chats, getMessages, setMessages,
// clearChat}) so call sites (App.jsx) need no changes.
export function useChats() {
  const snapshot = useSyncExternalStore(chatStore.subscribe, chatStore.getSnapshot, chatStore.getSnapshot)
  const getMessages = useCallback((modelKey) => chatStore.getMessages(modelKey), [])
  const setMessages = useCallback((modelKey, updater) => chatStore.setMessages(modelKey, updater), [])
  const clearChat = useCallback((modelKey) => chatStore.clearChat(modelKey), [])
  return { chats: snapshot.chats, getMessages, setMessages, clearChat }
}
