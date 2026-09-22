export type AuthConfig = {
    mode: 'external' | 'open';
    base: string | null;
    /** Did GET /api/auth/config actually answer? false ⇒ backend unreachable. */
    reachable: boolean;
};
export declare function getAuthConfig(): Promise<AuthConfig>;
export declare function getAuthBase(): Promise<string>;
