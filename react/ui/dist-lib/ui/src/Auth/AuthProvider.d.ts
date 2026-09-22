import { default as React } from 'react';
export type AuthUser = {
    username?: string;
    local?: boolean;
    [key: string]: unknown;
};
export type AuthState = {
    status: 'checking';
} | {
    status: 'guest';
} | {
    status: 'authed';
    user: AuthUser;
};
/** Result of a write action that may carry a server-supplied error message. */
export interface AuthResult {
    ok: boolean;
    error?: string;
}
/** Paths on the auth service, relative to its base. Override individually. */
export interface AuthEndpoints {
    me: string;
    login: string;
    logout: string;
    register: string;
    changePassword: string;
}
export interface AuthContextValue {
    state: AuthState;
    /** Resolved auth mode ('open' = no login wall, 'external' = login service). */
    mode: 'open' | 'external';
    /** false ⇒ the API instance is unreachable (backendless front door). */
    reachable: boolean;
    /** Re-check the current session (GET /me). */
    refresh: () => Promise<void>;
    /** Returns true on success. */
    signIn: (username: string, password: string) => Promise<boolean>;
    signOut: () => Promise<void>;
    signUp: (input: {
        username: string;
        email?: string;
        password: string;
    }) => Promise<AuthResult>;
    changePassword: (input: {
        currentPassword: string;
        newPassword: string;
    }) => Promise<AuthResult>;
}
export interface AuthProviderProps {
    children?: React.ReactNode;
    /** Auth service origin. If omitted, resolved from GET /api/auth/config. */
    base?: string;
    /** Force a mode. If omitted, resolved from config. */
    mode?: 'open' | 'external';
    /** Override one or more endpoint paths. */
    endpoints?: Partial<AuthEndpoints>;
    /** Custom fetch (e.g. to inject headers). Defaults to global fetch. */
    fetch?: typeof fetch;
    /** Credentials mode for auth requests. Defaults to 'include' (cookie auth). */
    credentials?: RequestCredentials;
}
export declare function AuthProvider({ children, base, mode, endpoints, fetch: fetchProp, credentials, }: AuthProviderProps): React.JSX.Element;
export declare function useAuth(): AuthContextValue;
