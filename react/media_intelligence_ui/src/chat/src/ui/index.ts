/*
 * ui/index.ts — barrel for the chat shell components.
 *
 * The old ChatDisplay / ChatInputPanel / ChatControls / RenderChat are
 * not re-exported because they no longer exist in this version. Delete
 * those files from disk; nothing imports them.
 */

export { default as Navbar }           from "./Navbar";
export { default as Sidebar }          from "./Sidebar";
export { default as ThreadHeader }     from "./ThreadHeader";
export { default as Thread }           from "./Thread";
export { default as MessageTurn }      from "./MessageTurn";
export { default as ToolRunTurn }      from "./ToolRunTurn";
export { default as ToolTray }         from "./ToolTray";
export { default as ChatResponse }     from "./ChatResponse";
export { default as ChatTurnActions }  from "./ChatTurnActions";
export { default as Composer }         from "./Composer";
export { default as ComposerControls } from "./ComposerControls";
export { default as Disclaimer }       from "./Disclaimer";
export { Icon }                        from "./Icons";
export type { IconName }               from "./Icons";
