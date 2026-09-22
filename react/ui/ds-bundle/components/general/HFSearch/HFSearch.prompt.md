HFSearch from @hugpy/ui. Use via `window.HugpyUI.HFSearch` (bundle loaded from the root `_ds_bundle.js`). Wrap the tree in `<DemoProvider>` (full provider chain in README.md — components read theme/i18n from that context).

## Props

```ts
interface HFSearchProps {
  onJobStarted: any;
  onCancelJob: any;
  onRetryJob: any;
  pendingByHub: any;
  jobsByHub: any;
  expanded: any;
  onToggleExpanded: any;
  embedded?: boolean;
}
```
