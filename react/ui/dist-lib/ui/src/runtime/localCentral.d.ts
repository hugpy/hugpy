export function isDemoHost(): boolean;
export function normalizeCentral(u: any): any;
/** The central the visitor last pointed at, else the localhost default. */
export function readSavedCentral(): any;
/**
 * Probe whether a hugpy central is listening at `base`. Resolves `true` if a
 * server answers (even without CORS headers), `false` on connection-refused,
 * abort/timeout, or a blank URL. See the module header for why this is `no-cors`.
 */
export function probeLocalCentral(raw: any, { timeoutMs }?: {
    timeoutMs?: number;
}): Promise<boolean>;
export const DEFAULT_CENTRAL: "http://localhost:7002";
export const CENTRAL_LS_KEY: "hugpy.centralUrl";
