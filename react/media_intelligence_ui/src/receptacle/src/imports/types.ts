export type InputMode = 'text' | 'url' | 'file';

export interface UploadedFileRef {
  name: string;
  /**
   * Opaque, session-scoped file handle issued by the server. Until the server emits
   * real ids it falls back to the upload path, but the CLIENT treats it as an opaque
   * token — it is never rendered or path-parsed. (Phase 6: closing the IDOR fully
   * requires the server to resolve id → path under session ownership.)
   */
  id: string;
  type?: string;
  size?: number;
}

export interface MediaInputValue {
  inputMode: InputMode;
  text: string;
  url: string;
  files: File[];
  uploadedFiles: UploadedFileRef[];
  selectedIds: string[];
}

export interface MediaInputProps {
  value: MediaInputValue;
  onChange: (next: MediaInputValue) => void;
}