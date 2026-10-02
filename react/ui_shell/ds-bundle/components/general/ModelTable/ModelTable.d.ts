import * as React from 'react';

/**
 * ModelTable — from @hugpy/ui@0.2.0.
 */
export interface ModelTableProps {
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

export declare const ModelTable: React.ComponentType<ModelTableProps>;
