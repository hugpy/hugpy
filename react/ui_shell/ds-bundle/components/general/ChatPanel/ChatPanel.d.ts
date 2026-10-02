import * as React from 'react';

/**
 * ChatPanel — from @hugpy/ui@0.2.0.
 */
export interface ChatPanelProps {
  modelKey: any;
  model: any;
  onClose: any;
  messages?: any[];
  setMessages: any;
}

export declare const ChatPanel: React.ComponentType<ChatPanelProps>;
