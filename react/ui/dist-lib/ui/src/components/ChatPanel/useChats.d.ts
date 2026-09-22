export function useChats(): {
    chats: any;
    getMessages: (modelKey: any) => any;
    setMessages: (modelKey: any, updater: any) => void;
    clearChat: (modelKey: any) => void;
};
