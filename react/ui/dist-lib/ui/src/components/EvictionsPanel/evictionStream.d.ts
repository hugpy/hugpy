/** Force a reconnect (the panel's Reconnect button). */
export function reconnect(): void;
/**
 * Subscribe to the shared eviction stream.
 *
 * `paused` freezes THIS view's snapshot without touching the stream or the
 * other view: the store keeps ingesting, and `held` counts what arrived while
 * frozen, so resuming lands everything in order. Pause is per-view on purpose —
 * pausing the Models-tab feed to read a row must not blind the Evictions tab.
 */
export function useEvictionRuns({ paused }?: {
    paused?: boolean;
}): {
    runs: any[];
    conn: string;
    error: any;
    held: number;
    reconnect: () => void;
};
/** Test/diagnostic reset — drops everything and detaches. */
export function _resetForTests(): void;
