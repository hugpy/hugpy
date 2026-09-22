export function GpuChips({ gpus }: {
    gpus: any;
}): import("react").JSX.Element;
export function PidRegistry({ worker, onEvict }: {
    worker: any;
    onEvict: any;
}): import("react").JSX.Element;
export function ResidentList({ items, loadedSet, worker, resource, emptyLabel, sizeByKey }: {
    items: any;
    loadedSet: any;
    worker: any;
    resource: any;
    emptyLabel: any;
    sizeByKey: any;
}): import("react").JSX.Element;
export function VramReconcile({ worker, gpuRes, comfy, sizeByKey }: {
    worker: any;
    gpuRes: any;
    comfy: any;
    sizeByKey: any;
}): import("react").JSX.Element;
export function ResourceChip({ icon, name, used, total, count, active, onClick, disabled, physicalTotal, bar, extraTerm }: {
    icon: any;
    name: any;
    used: any;
    total: any;
    count: any;
    active: any;
    onClick: any;
    disabled: any;
    physicalTotal: any;
    bar: any;
    extraTerm: any;
}): import("react").JSX.Element;
export function ResourceStrip({ worker, models, onApproveEvictions, onEvict }: {
    worker: any;
    models: any;
    onApproveEvictions: any;
    onEvict: any;
}): import("react").JSX.Element;
export const ESTIMATED_VRAM_TITLE: "estimated from declared placement \u00D7 file size \u2014 not measured";
