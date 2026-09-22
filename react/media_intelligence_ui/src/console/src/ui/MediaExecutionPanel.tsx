import {styles} from "./../imports";
import type {
  MediaInputValue,
  MediaKind,
  Operation,
  PageSpec,
} from "./../imports/pages/pageSpec";
import type { PageValues } from "./../imports/page/submitPage";
import {UtilityPage} from "./../imports/page/UtilityPage";

type MediaExecutionPanelProps = {
  selectedSpec?: PageSpec;
  source: MediaInputValue;
  media: MediaKind;
  operation: Operation | "any";
  onJumpToInput: () => void;
  values: PageValues;
};

export function MediaExecutionPanel({
  selectedSpec,
  source,
  media,
  operation,
  onJumpToInput,
  values,
}: MediaExecutionPanelProps) {
  if (!selectedSpec) {
    return (
      <div className={styles.emptyState}>
        No tools match {media}
        {operation !== "any" ? ` / ${operation}` : ""}.
      </div>
    );
  }

  return (
    <div className={styles.executionArea}>
      <UtilityPage
        key={selectedSpec.key}
        spec={selectedSpec}
        input={source}
        onJumpToInput={onJumpToInput}
        media={media}
        operation={operation}
        values={values}
      />
    </div>
  );
}