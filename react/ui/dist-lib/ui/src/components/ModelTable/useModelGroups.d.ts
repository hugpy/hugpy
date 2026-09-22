export function useModelGroups(): {
    byModel: Map<any, any>;
    multiMember: any[];
    setTick: (groupKey: any, tick: any, next: any) => Promise<boolean>;
    notice: any;
    clearNotice: () => void;
    offHint: string;
    refresh: () => Promise<void>;
    enabled: boolean;
    source: string;
    groups: any[];
    loading: boolean;
    error: any;
};
/**
 * The per-worker verdict chips for one member: what it does on each box, or
 * why it lost there. Pure — the backend already decided; this only formats.
 */
export function memberVerdicts(group: any, modelKey: any): {
    worker: string;
    tone: string;
    text: string;
    title: any;
}[];
export const TICKS: string[];
export namespace TICK_HELP {
    let quality: string;
    let speed: string;
    let priority: string;
}
