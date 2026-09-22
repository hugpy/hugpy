export default function HFSearch({ models, onJobStarted, onCancelJob, onRetryJob, pendingByHub, jobsByHub, expanded, onToggleExpanded, embedded }: {
    models?: any[];
    onJobStarted: any;
    onCancelJob: any;
    onRetryJob: any;
    pendingByHub: any;
    jobsByHub: any;
    expanded: any;
    onToggleExpanded: any;
    embedded?: boolean;
}): import("react").JSX.Element;
