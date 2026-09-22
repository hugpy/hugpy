import * as React from 'react';

/**
 * HFSearch — from @hugpy/ui@0.2.0.
 */
export interface HFSearchProps {
  onJobStarted: any;
  onCancelJob: any;
  onRetryJob: any;
  pendingByHub: any;
  jobsByHub: any;
  expanded: any;
  onToggleExpanded: any;
  embedded?: boolean;
}

export declare const HFSearch: React.ComponentType<HFSearchProps>;
