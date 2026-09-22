import { useEffect, useMemo, useRef, useState } from "react";

import { MediaInputReceptacle } from "./../receptacle";
import type {
  InputMode,
  MediaInputValue,
} from "./../receptacle";

import {
  listPages,
  type MediaKind,
  type Operation,
} from "./src/imports";

import {
  MEDIA_BY_INPUT_MODE,
  inferFileMedia,
  asArray,
} from "./src/utilities";

import "./src/imports/pages/pagesBuiltin";
import "./src/imports/chain/chainsBuiltin";

import {
  MediaInputDropdowns,
  MediaInputButtons,
  MediaExecutionPanel,
} from "./src";

import { initialValues } from "./src/imports/page/UtilityPage/UtilityPage";
import type { PageValues } from "./src/imports/page/submitPage";

import { styles } from "./src";

const DEFAULT_SOURCE: MediaInputValue = {
  inputMode: "text",
  text: "",
  url: "",
  files: [],
  uploadedFiles: [],
  selectedIds: [],
};

export default function HugpyConsole() {
  const receptacleRef = useRef<HTMLDivElement | null>(null);

  const allPages = useMemo(() => listPages(), []);

  const [source, setSource] = useState<MediaInputValue>(DEFAULT_SOURCE);
  const [media, setMedia] = useState<MediaKind>("text");
  const [operation, setOperation] = useState<Operation | "any">("any");
  const [selectedKey, setSelectedKey] = useState("");

  const allowedMedia = useMemo(() => {
    return MEDIA_BY_INPUT_MODE[source.inputMode as InputMode] ?? [];
  }, [source.inputMode]);

  useEffect(() => {
    const fallbackMedia = allowedMedia[0];

    if (fallbackMedia && !allowedMedia.includes(media)) {
      setMedia(fallbackMedia);
      setOperation("any");
    }
  }, [allowedMedia, media]);

  useEffect(() => {
    if (source.inputMode !== "file") return;

    const inferred = inferFileMedia(source.files, source.uploadedFiles);

    if (inferred && allowedMedia.includes(inferred)) {
      setMedia(inferred);
      setOperation("any");
    }
  }, [
    source.files,
    source.uploadedFiles,
    source.inputMode,
    allowedMedia,
  ]);

  const operationsForMedia = useMemo(() => {
    const ops = allPages
      .filter((page) => page.accepts.includes(media))
      .flatMap((page) => asArray(page.produces));

    return Array.from(new Set(ops)).sort();
  }, [allPages, media]);

  const filteredPages = useMemo(() => {
    return allPages.filter((page) => {
      if (!page.accepts.includes(media)) return false;
      if (operation === "any") return true;

      return asArray(page.produces).includes(operation);
    });
  }, [allPages, media, operation]);

  useEffect(() => {
    const selectedStillExists = filteredPages.some(
      (page) => page.key === selectedKey,
    );

    if (!selectedStillExists) {
      setSelectedKey(filteredPages[0]?.key ?? "");
    }
  }, [filteredPages, selectedKey]);

  const selectedSpec = useMemo(() => {
    return filteredPages.find((page) => page.key === selectedKey);
  }, [filteredPages, selectedKey]);

  // The selected tool's option values live here (alongside the selected spec +
  // operation) so they can render in the operation dropdown row. Reset to the
  // spec's defaults whenever the selected tool changes.
  const [toolValues, setToolValues] = useState<PageValues>(() =>
    selectedSpec ? initialValues(selectedSpec) : {},
  );

  useEffect(() => {
    setToolValues(selectedSpec ? initialValues(selectedSpec) : {});
  }, [selectedSpec?.key]);

  function setToolField(name: string, value: PageValues[string]) {
    setToolValues((prev) => ({ ...prev, [name]: value }));
  }

  function jumpToInput() {
    receptacleRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  }

return (
  <main className={styles.page}>
    <div ref={receptacleRef} className={styles.inputConsoleDeck}>
      <MediaInputReceptacle value={source} onChange={setSource} />

      <MediaInputButtons
        filteredPages={filteredPages}
        selectedKey={selectedKey}
        setSelectedKey={setSelectedKey}
      />

      <MediaInputDropdowns
        source={source}
        media={media}
        setMedia={setMedia}
        operation={operation}
        setOperation={setOperation}
        allowedMedia={allowedMedia}
        operationsForMedia={operationsForMedia}
        selectedSpec={selectedSpec}
        values={toolValues}
        onChangeField={setToolField}
      />

      <MediaExecutionPanel
        selectedSpec={selectedSpec}
        source={source}
        media={media}
        operation={operation}
        onJumpToInput={jumpToInput}
        values={toolValues}
      />
    </div>
  </main>
);
}