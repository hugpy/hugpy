import {styles} from './../../imports';
interface LocalFileListProps {
  files: File[];
}

export function LocalFileList({ files }: LocalFileListProps) {
  if (!files.length) return null;

  return (
    <ul
      className={styles.fileList}
      onClick={(event) => event.stopPropagation()}
    >
      {files.map((file) => (
        <li key={`${file.name}-${file.size}`}>
          <div>
            <strong>{file.name}</strong>
            <small>local only</small>
          </div>

          <span>{(file.size / 1024 / 1024).toFixed(2)} MB</span>
        </li>
      ))}
    </ul>
  );
}
