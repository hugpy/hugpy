import {styles,type UploadedFileRef} from './../../imports';

interface UploadedFileListProps {
  uploadedFiles: UploadedFileRef[];
  selectedIds: string[];
  allSelected: boolean;
  onToggleSelected: (id: string) => void;
  onSelectAll: () => void;
  onClearSelection: () => void;
}

export function UploadedFileList({
  uploadedFiles,
  selectedIds,
  allSelected,
  onToggleSelected,
  onSelectAll,
  onClearSelection,
}: UploadedFileListProps) {
  if (!uploadedFiles.length) return null;

  return (
    <div
      className={styles.fileListWrap}
      onClick={(event) => event.stopPropagation()}
    >
      <div className={styles.selectionToolbar}>
        <span>
          {selectedIds.length} / {uploadedFiles.length} selected
        </span>

        <div className={styles.selectionActions}>
          <button
            type="button"
            className={styles.linkBtn}
            onClick={allSelected ? onClearSelection : onSelectAll}
          >
            {allSelected ? "Clear" : "Select all"}
          </button>
        </div>
      </div>

      <ul className={styles.fileList}>
        {uploadedFiles.map((file) => {
          const checked = selectedIds.includes(file.id);

          return (
            <li
              key={file.id}
              className={checked ? styles.fileRowSelected : ""}
              onClick={() => onToggleSelected(file.id)}
            >
              <label
                className={styles.fileRowLabel}
                onClick={(event) => event.stopPropagation()}
              >
                <input
                  type="checkbox"
                  checked={checked}
                  onChange={() => onToggleSelected(file.id)}
                />

                <div>
                  <strong>{file.name}</strong>
                  {/* Phase 6: never render the server path. Show size, not location. */}
                  {file.size != null && (
                    <small>{Math.max(1, Math.round(file.size / 1024))} KB</small>
                  )}
                </div>
              </label>

              <span>{file.type ?? "file"}</span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
