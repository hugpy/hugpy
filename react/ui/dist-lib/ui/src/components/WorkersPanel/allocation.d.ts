export function spillToMode(spill: any): {
    mode: string;
    gib: string;
    ram: string;
    threads: string;
    gpuBand: string;
    ramBand: string;
    ctxPct: string;
    ctxBand: string;
    priority: string;
};
export function modeToSpill(mode: any, gib: any, ram: any, threads: any, gpuBand: any, ramBand: any, ctxPct: any, ctxBand: any, priority: any): {
    gpu_mem_gib: number;
    cpu_mem_gib: number;
    threads: number;
    gpu_mem_gib_deviation_pct: number;
    cpu_mem_gib_deviation_pct: number;
    ctx_pct: number;
    ctx_deviation_pct: number;
    priority: number;
} | {
    n_gpu_layers: number;
} | {
    n_gpu_layers: string;
} | {
    n_gpu_layers?: undefined;
};
export function allocIsGgufOnly(spill: any): boolean;
export function spillLabel(spill: any): string;
export function workerCapacity(worker: any, which: any): {
    bytes: number;
    gib: any;
    basis: string;
} | {
    bytes: any;
    gib: number;
    basis: string;
};
export function deriveAllocMode(spill: any): "gpu-only" | "ram-only" | "max-gpu" | "max-ram" | "explicit";
export function allocModeLabel(mode: any): any;
export function resolvedSeatMode(ngl: any, total: any, gpuPct: any): {
    mode: string;
    split: string;
};
export function fmtGiB(bytes: any): string;
export function allocDisableReason(value: any, ctx: any, engineGguf: any): string;
export const EXPLICIT_BUDGET_KEYS: string[];
