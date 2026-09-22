import React from "react";
import { ChatDisplay, type ChatEntry } from "./../chat";
import physicsChatJson from "./data/chats.json";
import claudeChatsJson from "./data/chats_claude.json";
import {styles} from "./../chat";

const claudeChats = claudeChatsJson as ChatEntry[];
const physicsChat = physicsChatJson as ChatEntry[];

export function ClaudeChat() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-blue-50 flex justify-center items-center">
      <ChatDisplay ref={null} chats={claudeChats} modelLabel={"chat gpt"}/>
    </div>
  );
}

export function PhysicsChat() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-blue-50 flex justify-center items-center">
      <ChatDisplay ref={null} chats={physicsChat} modelLabel={"claude"}/>
    </div>
  );
}