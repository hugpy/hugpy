import {type InputMode,styles } from "./../imports";

interface ModeTabsProps {
  inputMode: InputMode;
  onChange: (mode: InputMode) => void;
}

const MODES: InputMode[] = ["text", "url", "file"];

export function ModeTabs({ inputMode, onChange }: ModeTabsProps) {
  return (
    <div className={styles.modeTabs}>
      {MODES.map((mode) => (
        <button
          key={mode}
          type="button"
          className={`${styles.modeTab} ${
            inputMode === mode ? styles.active : ""
          }`}
          onClick={() => onChange(mode)}
        >
          {mode.toUpperCase()}
        </button>
      ))}
    </div>
  );
}
