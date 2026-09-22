import { type MediaInputProps,styles,useMediaInputReceptacle } from "./imports";
import { ModeTabs } from "./ui/ModeTabs";
import { TextInputStage } from "./ui/stages/TextInputStage";
import { UrlInputStage } from "./ui/stages/UrlInputStage";
import { FileInputStage } from "./ui/stages/FileInputStage";

export default function MediaInputReceptacle({
  value,
  onChange,
}: MediaInputProps) {
  const receptacle = useMediaInputReceptacle({ value, onChange });

  return (
    <section className={styles.receptacle}>
      <ModeTabs
        inputMode={value.inputMode}
        onChange={(inputMode) => receptacle.patch({ inputMode })}
      />

      <div className={styles.inputStage}>
        {value.inputMode === "text" && (
          <TextInputStage
            value={value.text}
            onChange={(text) => receptacle.patch({ text })}
          />
        )}

        {value.inputMode === "url" && (
          <UrlInputStage
            value={value.url}
            onChange={(url) => receptacle.patch({ url })}
          />
        )}

        {value.inputMode === "file" && (
          <FileInputStage
            value={value}
            dragActive={receptacle.dragActive}
            uploading={receptacle.uploading}
            uploadError={receptacle.uploadError}
            allSelected={receptacle.allSelected}
            onSetDragActive={receptacle.setDragActive}
            onSetFiles={receptacle.setFiles}
            onToggleSelected={receptacle.toggleSelected}
            onSelectAll={receptacle.selectAll}
            onClearSelection={receptacle.clearSelection}
          />
        )}
      </div>
    </section>
  );
}
