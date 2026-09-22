import { useParams } from "react-router-dom";
import { getPage } from "./../imports/pages/pagesRegistry";
import {UtilityPage} from "./../imports/page/UtilityPage";

import "./../imports/pages/pagesBuiltin";
import "./../imports/chain/chainsBuiltin";

import type {
  MediaInputValue,
  MediaKind,
  Operation,
} from "./../imports/pages/pageSpec";

interface UtilityRouteProps {
  input?: MediaInputValue;
  media?: MediaKind;
  operation?: Operation | "any";
  onJumpToInput?: () => void;
}

function getDefaultInput(): MediaInputValue {
  return {
    inputMode: "text",
    text: "",
    url: "",
    files: [],
    uploadedFiles: [],
    selectedIds: [],
  };
}

export default function UtilityRoute({
  input = getDefaultInput(),
  media,
  operation = "any",
  onJumpToInput = () => {},
}: UtilityRouteProps) {
  const { "*": key } = useParams();

  if (!key) return <div>404</div>;

  try {
    const spec = getPage(key);

    return (
      <UtilityPage
        spec={spec}
        input={input}
        media={media ?? spec.accepts[0] ?? "text"}
        operation={operation}
        onJumpToInput={onJumpToInput}
      />
    );
  } catch {
    return <div>Unknown utility: {key}</div>;
  }
}