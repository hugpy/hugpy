export type ApiErrorPayload = {
    error?: string;
    detail?: string;
    message?: string;
    [key: string]: unknown;
};
export type UploadFileResponse = {
    path: string;
    name: string;
    size: number;
    [key: string]: unknown;
};
export declare function fetchJson<T = unknown>(url: string, options?: RequestInit): Promise<T>;
export declare function uploadFile(file: File): Promise<UploadFileResponse>;
