import {styles } from "./../../imports";

interface TextInputStageProps {
  value: string;
  onChange: (text: string) => void;
}

export function TextInputStage({ value, onChange }: TextInputStageProps) {
  return (
    <textarea
      className={styles.textBox}
      value={value}
      onChange={(event) => onChange(event.target.value)}
      placeholder="Paste or write text to analyze..."
      rows={8}
    />
  );
}
