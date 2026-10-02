ChatPanel from @hugpy/ui. Use via `window.HugpyUI.ChatPanel` (bundle loaded from the root `_ds_bundle.js`). Wrap the tree in `<DemoProvider>` (full provider chain in README.md — components read theme/i18n from that context).

## Props

```ts
interface ChatPanelProps {
  modelKey: any;
  model: any;
  onClose: any;
  messages?: any[];
  setMessages: any;
}
```
