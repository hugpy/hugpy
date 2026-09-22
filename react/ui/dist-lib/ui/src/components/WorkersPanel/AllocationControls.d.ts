export function SliderReadout({ text, value, onCommit, title }: {
    text: any;
    value: any;
    onCommit: any;
    title: any;
}): import("react").JSX.Element;
export function PlainSlider({ label, title, value, onChange, min, max, step, formatValue, clampMax, className }: {
    label: any;
    title: any;
    value: any;
    onChange: any;
    min: any;
    max: any;
    step?: number;
    formatValue: any;
    clampMax?: boolean;
    className?: string;
}): import("react").JSX.Element;
export function BudgetInput({ label, title, unit, value, onChange, cap, fallbackMax, modelCap, basis, onBasisChange }: {
    label: any;
    title: any;
    unit: any;
    value: any;
    onChange: any;
    cap: any;
    fallbackMax?: number;
    modelCap?: any;
    basis?: string;
    onBasisChange?: any;
}): import("react").JSX.Element;
export function AllocControl({ spill, onApply, onCancel, applyLabel, engineGguf, worker, need, bulkKeys, getModelBytes }: {
    spill: any;
    onApply: any;
    onCancel: any;
    applyLabel?: string;
    engineGguf?: any;
    worker?: any;
    need?: any;
    bulkKeys?: any;
    getModelBytes?: any;
}): import("react").JSX.Element;
export function ExplicitPanel({ spill, worker, need, onApply, onCancel }: {
    spill: any;
    worker: any;
    need: any;
    onApply: any;
    onCancel: any;
}): import("react").JSX.Element;
export function AllocModeMenu({ mode, spill, worker, need, engineGguf, feasible, feasibleCtx, derivedMode, anchorRef, onPick, onApplyExplicit, onRevertDerived, onClose }: {
    mode: any;
    spill: any;
    worker: any;
    need: any;
    engineGguf: any;
    feasible: any;
    feasibleCtx: any;
    derivedMode: any;
    anchorRef: any;
    onPick: any;
    onApplyExplicit: any;
    onRevertDerived: any;
    onClose: any;
}): import("react").JSX.Element;
export function BulkExplicitPanel({ bulkKeys, getModelBytes, onApply, onCancel }: {
    bulkKeys: any;
    getModelBytes: any;
    onApply: any;
    onCancel: any;
}): import("react").JSX.Element;
export function BulkAllocControl({ count, bulkKeys, getModelBytes, onApply, onCancel }: {
    count: any;
    bulkKeys: any;
    getModelBytes: any;
    onApply: any;
    onCancel: any;
}): import("react").JSX.Element;
