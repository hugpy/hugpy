ModelTable from @hugpy/ui. Use via `window.HugpyUI.ModelTable` (bundle loaded from the root `_ds_bundle.js`). Wrap the tree in `<DemoProvider>` (full provider chain in README.md — components read theme/i18n from that context).

## Props

```ts
interface ModelTableProps {
  models: any;
  jobsByModel: any;
  activeChat: any;
  onDownload: any;
  onChat: any;
  onDelete: any;
  onPrune: any;
  onSetMedia: any;
  onCancel: any;
  onRetry: any;
  workers?: any[];
  onAssignWorker: any;
  onProbeWorker: any;
}
```
