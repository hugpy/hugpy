import { useRef } from "react";
import { styles, type MediaInputValue } from "./../../imports";
import { UploadedFileList, LocalFileList } from "./../files";

interface FileInputStageProps {
  value: MediaInputValue;
  dragActive: boolean;
  uploading: boolean;
  uploadError: string;
  allSelected: boolean;
  onSetDragActive: (active: boolean) => void;
  onSetFiles: (files: globalThis.FileList | File[]) => void;
  onToggleSelected: (id: string) => void;
  onSelectAll: () => void;
  onClearSelection: () => void;
}

export function FileInputStage({
  value,
  dragActive,
  uploading,
  uploadError,
  allSelected,
  onSetDragActive,
  onSetFiles,
  onToggleSelected,
  onSelectAll,
  onClearSelection,
}: FileInputStageProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  return (
    <div>
      {/* Hidden input + interactive file list are SIBLINGS of the button, never nested
          inside it — a role="button" must not contain other interactive controls. */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        tabIndex={-1}
        aria-hidden="true"
        className={styles.hiddenInput}
        onChange={(event) => onSetFiles(event.target.files ?? [])}
      />

      <div
        className={`${styles.dropZone} ${dragActive ? styles.dragActive : ""}`}
        role="button"
        tabIndex={0}
        aria-label="Upload files — drop here, or press Enter to browse"
        onDragOver={(event) => {
          event.preventDefault();
          onSetDragActive(true);
        }}
        onDragLeave={() => onSetDragActive(false)}
        onDrop={(event) => {
          event.preventDefault();
          onSetDragActive(false);
          onSetFiles(event.dataTransfer.files);
        }}
        onClick={() => fileInputRef.current?.click()}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            fileInputRef.current?.click();
          }
        }}
      >
        <div className={styles.dropTitle}>
          {uploading ? "Uploading..." : "Drop files here"}
        </div>

        <div className={styles.dropSubtext}>
          {uploading ? "Sending to HugPy upload service" : "or click to browse"}
        </div>
      </div>

      {uploadError && <div className={styles.uploadError}>{uploadError}</div>}

      <UploadedFileList
        uploadedFiles={value.uploadedFiles}
        selectedIds={value.selectedIds}
        allSelected={allSelected}
        onToggleSelected={onToggleSelected}
        onSelectAll={onSelectAll}
        onClearSelection={onClearSelection}
      />

      {!value.uploadedFiles.length && <LocalFileList files={value.files} />}
    </div>
  );
}