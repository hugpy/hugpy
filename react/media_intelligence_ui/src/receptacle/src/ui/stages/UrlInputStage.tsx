import {styles } from "./../../imports";
interface UrlInputStageProps {
  value: string;
  onChange: (url: string) => void;
}

export function UrlInputStage({ value, onChange }: UrlInputStageProps) {
  return (
    <input
      className={styles.urlBar}
      value={value}
      onChange={(event) => onChange(event.target.value)}
      // eslint-disable-next-line no-restricted-syntax -- UI placeholder text, not a service URL
      placeholder="https://example.com/video-or-page"
      type="url"
    />
  );
}
