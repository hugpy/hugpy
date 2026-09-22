export function modelTask(m: any): any;
export function modelTasks(m: any): any[];
export default function ModelTable({ models, jobsByModel, activeChat, onDownload, onChat, onDelete, onPrune, onSetMedia, onSetMediaDefault, onCancel, onRetry, workers, onAssignWorker, onProbeWorker, onRefresh, }: {
    models: any;
    jobsByModel: any;
    activeChat: any;
    onDownload: any;
    onChat: any;
    onDelete: any;
    onPrune: any;
    onSetMedia: any;
    onSetMediaDefault: any;
    onCancel: any;
    onRetry: any;
    workers?: any[];
    onAssignWorker: any;
    onProbeWorker: any;
    onRefresh: any;
}): import("react").JSX.Element;
