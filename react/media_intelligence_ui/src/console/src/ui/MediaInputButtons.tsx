import type { Dispatch, SetStateAction } from "react";
import type {
  PageSpec,
} from "./../imports/pages/pageSpec";
import {styles} from "./../imports";


type MediaInputButtonsProps = {
  filteredPages: PageSpec[];
  selectedKey: string;
  setSelectedKey: Dispatch<SetStateAction<string>>;
};

export function MediaInputButtons({
  filteredPages,
  selectedKey,
  setSelectedKey,
}: MediaInputButtonsProps) {
  return (
    <section className={styles.toolGrid}>
      {filteredPages.map((page) => (
        <button
          key={page.key}
          type="button"
          className={`${styles.toolCard} ${
            selectedKey === page.key ? styles.selectedTool : ""
          }`}
          onClick={() => setSelectedKey(page.key)}
        >
          <h3>{page.title}</h3>
          <p>{page.category}</p>
        </button>
      ))}
    </section>
  );
}